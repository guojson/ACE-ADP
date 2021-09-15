import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

from tf_utils.bert_modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint, layer_norm
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.layers.python.layers import initializers
from tf_utils import rnncell as rnn
from utils import get_shape


class Model:

    def __init__(self, config):
        self.config = config
        # 喂入模型的数据占位符
        self.input_x_word = tf.placeholder(tf.int32, [None, None], name="input_x_word")
        self.input_x_len = tf.placeholder(tf.int32, name='input_x_len')
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')
        self.input_relation = tf.placeholder(tf.int32, [None, None], name='input_relation')  # 实体NER的真实标签

        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')

        # BERT Embedding
        self.bert_embedding(bert_init=True)
        output_layer = self.word_embedding

        # 超参数设置
        self.relation_num = self.config.relation_num[self.config.dataset]
        self.initializer = initializers.xavier_initializer()
        self.lstm_dim = self.config.lstm_dim
        self.embed_dense_dim = self.config.embed_dense_dim
        self.dropout = self.config.dropout
        self.model_type = self.config.model_type
        print('Run Model Type:', self.model_type)


        self.filter_width = 3
        self.num_filter = self.lstm_dim
        self.embedding_dim = self.embed_dense_dim
        self.repeat_times = 4
        self.cnn_output_width = 0

        # CRF超参数
        used = tf.sign(tf.abs(self.input_x_word))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.input_x_word)[0]
        self.num_steps = tf.shape(self.input_x_word)[-1]

        self.loss = self.computeLoss(output_layer, name='nomal')
        if self.config.adversarial:
            embedding_input = output_layer
            raw_perturb = tf.gradients(self.loss, embedding_input)[0]  # [batch, L, dim]
            normalized_per = tf.nn.l2_normalize(raw_perturb, axis=[1, 2])
            perturb = self.config.alpha * tf.sqrt(tf.cast(tf.shape(output_layer)[2], tf.float32)) * tf.stop_gradient(
                normalized_per)
            perturb_inputs = embedding_input + perturb
            ad_loss = self.computeLoss(perturb_inputs, name='ad_module')
            self.loss += ad_loss


    def computeLoss(self, input, name):
        # 添加gated cnn 层
        inputs = tf.nn.dropout(input, self.dropout)
        if self.model_type == 'bilstm':
            # inputs = tf.nn.dropout(inputs, self.dropout)
            lstm_outputs = self.biLSTM_layer(inputs, self.lstm_dim, self.lengths, name)
            self.logits = self.project_layer(lstm_outputs, name)
        else:
            raise KeyError
        # 计算损失
        loss = self.loss_layer(self.logits, self.lengths, name)
        return loss

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.name_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.name_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths, scope=name)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.name_scope("project" if not name else name):
            with tf.name_scope("hidden"):
                W = tf.get_variable("HW" + name, shape=[self.lstm_dim * 2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("Hb" + name, shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.name_scope("logits"):
                W = tf.get_variable("LW" + name, shape=[self.lstm_dim, self.relation_num],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("Lb" + name, shape=[self.relation_num], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            logits = tf.reshape(pred, [-1, self.num_steps, self.relation_num], name='pred_logits')
            tf.add_to_collection('pred_logits', logits)
            return logits

    def loss_layer(self, project_logits, lengths, name=None):
        """
        计算CRF的loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.name_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.relation_num]),
                 tf.zeros(shape=[self.batch_size, 1, 1])],
                axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.relation_num * tf.ones([self.batch_size, 1]), tf.int32), self.input_relation], axis=-1)
            self.trans = tf.get_variable(
                name="transitions" + name,
                shape=[self.relation_num + 1, self.relation_num + 1],  # 1
                # shape=[self.relation_num, self.relation_num],  # 1
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                # tag_indices=self.input_relation,
                transition_params=self.trans,
                # sequence_lengths=lengths
                sequence_lengths=lengths + 1
            )  # + 1
            return tf.reduce_mean(-log_likelihood, name='loss')

    def bert_embedding(self, bert_init=True):
        """
        对BERT的Embedding降维
        :param bert_init:
        :return:
        """
        with tf.name_scope('embedding'):
            word_embedding = self.bert_embed(bert_init)
            print('self.embed_dense_dim:', self.config.embed_dense_dim)
            word_embedding = tf.layers.dense(word_embedding, self.config.embed_dense_dim, activation=tf.nn.relu)
            hidden_size = word_embedding.shape[-1].value
        self.word_embedding = word_embedding
        print(word_embedding.shape)
        self.output_layer_hidden_size = hidden_size

    def bert_embed(self, bert_init=True):
        """
        读取BERT的TF模型
        :param bert_init:
        :return:
        """
        bert_config_file = self.config.bert_config_file
        bert_config = BertConfig.from_json_file(bert_config_file)
        # batch_size, max_seq_length = get_shape_list(self.input_x_word)
        # bert_mask = tf.pad(self.input_mask, [[0, 0], [2, 0]], constant_values=1)  # tensor左边填充2列
        model = BertModel(
            config=bert_config,
            is_training=self.is_training,  # 微调
            input_ids=self.input_x_word,
            input_mask=self.input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=False)

        if self.config.use_origin_bert:
            final_hidden_states = model.get_sequence_output()  # 原生bert
            self.config.embed_dense_dim = 768
        else:
            layer_logits = []
            for i, layer in enumerate(model.all_encoder_layers):
                layer_logits.append(
                    tf.layers.dense(
                        layer, 1,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        name="layer_logit%d" % i
                    )
                )

            layer_logits = tf.concat(layer_logits, axis=2)  # 第三维度拼接
            layer_dist = tf.nn.softmax(layer_logits)
            seq_out = tf.concat([tf.expand_dims(x, axis=2) for x in model.all_encoder_layers], axis=2)
            pooled_output = tf.matmul(tf.expand_dims(layer_dist, axis=2), seq_out)
            pooled_output = tf.squeeze(pooled_output, axis=2)
            char_bert_outputs = pooled_output
            # char_bert_outputs = pooled_laRERyer[:, 1: max_seq_length - 1, :]  # [batch_size, seq_length, embedding_size]
            final_hidden_states = char_bert_outputs
            self.config.embed_dense_dim = 512

        tvars = tf.trainable_variables()
        init_checkpoint = self.config.bert_file  # './chinese_L-12_H-768_A-12/bert_model.ckpt'
        if self.config.use_fine_tuned:
            init_checkpoint = self.config.checkpoint_path
            tva = [x for x in tvars if 'bert' in x.name]  # BERT的参数
        else:
            tva = tvars

        # 选择属于BERT的模型进行初始化

        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tva, init_checkpoint)
        if bert_init:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tva:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            print("  name = {}, shape = {}{}".format(var.name, var.shape, init_string))
        print('init bert from checkpoint: {}'.format(init_checkpoint))
        return final_hidden_states