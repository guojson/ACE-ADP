import os
import pickle
import shutil
from time import *
import json
import tqdm
from config import Config
from model import Model
from optimization import create_optimizer
import numpy as np
from bert import tokenization
# import pandas as pd
from tensorflow.contrib.crf import viterbi_decode
import tensorflow as tf

from utils import get_logger, DataIterator

gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def train(train_iter, test_iter, dev_iter, config):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            model = Model(config)  # 读取模型结构图
            # 超参数设置
            global_step = tf.Variable(0, name='step', trainable=False)

            learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.decay_step,
                                                       config.decay_rate, staircase=True)
            normal_optimizer = tf.train.AdamOptimizer(learning_rate)  # 下接结构的学习率
            # normal_optimizer = tf.train.AdamOptimizer(config.learning_rate)

            all_variables = graph.get_collection('trainable_variables')
            normal_var_list = [x for x in all_variables if 'bert' not in x.name]  # 下接结构的参数
            logger.info('normal train variable num: {}'.format(len(normal_var_list)))
            normal_op = normal_optimizer.minimize(model.loss, global_step=global_step,
                                                  var_list=normal_var_list)  # 不参与下游训练
            num_batch = int(train_iter.num_records / config.batch_size * config.train_epoch)
            logger.info('num_batch:' + str(num_batch))
            logger.info('num_records:' + str(train_iter.num_records))

            # 对BERT微调
            if config.fine_tuning:
                embed_step = tf.Variable(0, name='step', trainable=False)
                word2vec_var_list = [x for x in all_variables if 'bert' in x.name]  # BERT的参数
                logger.info('bert train variable num: {}'.format(len(word2vec_var_list)))
                if word2vec_var_list:  # 对BERT微调
                    logger.info('word2vec trainable!!')
                    word2vec_op, embed_learning_rate, embed_step = create_optimizer(
                        model.loss, config.embed_learning_rate, num_train_steps=num_batch,
                        num_warmup_steps=int(num_batch * 0.05), use_tpu=False, variable_list=word2vec_var_list
                    )

                    train_op = tf.group(normal_op, word2vec_op)  # 组装BERT与下接结构参数
                else:
                    train_op = normal_op
            else:
                train_op = normal_op

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)
            if config.continue_training:
                logger.info('recover from: {}'.format(config.checkpoint_path))
                saver.restore(session, config.checkpoint_path)
            else:
                session.run(tf.global_variables_initializer())
            cum_step = 0
            F_max = 0
            dev_F_max = 0
            for i in range(config.train_epoch):  # 训练
                logger.info('epoch' + str(i))
                for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length, tokens_list, radical_ids_list, radical_lengths_list in tqdm.tqdm(
                        train_iter):

                    feed_dict = {
                        model.input_x_word: input_ids_list,
                        model.input_mask: input_mask_list,
                        model.input_relation: label_ids_list,
                        model.input_x_len: seq_length,
                        model.keep_prob: config.keep_prob,
                        model.is_training: config.is_training,
                    }
                    _, step, loss, lr = session.run(
                        fetches=[train_op,
                                 global_step,
                                 model.loss,
                                 learning_rate  # config.learning_rate
                                 ],
                        feed_dict=feed_dict)

                    if cum_step % 10 == 0:
                        format_str = 'step {}, loss {:.4f} lr {:.5f}'
                        logger.info(
                            format_str.format(
                                step, loss, lr)
                        )
                    cum_step += 1
                logger.info('traing finished')
                logger.info('dev start')
                accuracy, precision, recall, F = set_test(model, test_iter, session, config.out_dir, i, config)
                logger.info('dev finished')


                #此处可以根据需要以F1或test loss为评价标注设置early stopping,

                if F > F_max:  # 保存F1大于0的模型
                    logger.info('遇到最大值了，' + str(F))
                    F_max = F
                    if os.path.isdir(os.path.join(config.out_dir, 'model')):
                        if not config.fine_tuning:
                            shutil.rmtree(os.path.join(config.out_dir, 'model'))
                    saver.save(session, os.path.join(config.out_dir, 'model',
                                                     'model_{:.2f}_{:.2f}_{:.2f}_{:.2f}'.format(precision, recall, F,
                                                                                                i)), global_step=step)
                if config.use_test:
                    logger.info('test start')
                    dev_accuracy, dev_precision, dev_recall, dev_F = set_test(model, dev_iter, session,
                                                                              config.out_dir + '/dev/', i, config)
                    logger.info('test finished')
                    if dev_F_max < dev_F:
                        logger.info('遇到最大dev值了，' + str(dev_F))
                        dev_F_max = dev_F



def get_text_and_label(input_tokens_list, y_list):
    """
    还原每一条数据的文本的标签
    :return:
    """
    temp = []
    for batch_y_list in y_list:
        temp += batch_y_list
    y_list = temp

    y_label_list = []  # 标签
    for i, input_tokens in enumerate(input_tokens_list):
        ys = y_list[i]  # 每条数据对应的数字标签列表
        temp = []
        label_list = []
        for index, num in enumerate(ys):

            if num == 4 and len(temp) == 0:
                temp.append(input_tokens[index])
            elif num == 5 and len(temp) > 0:
                temp.append(input_tokens[index])
            elif len(temp) > 0:
                label = "".join(temp)
                if len(set(label)) > 1:  # 干掉单字重复情况
                    label_list.append("".join(temp))

                temp = []

        y_label_list.append(";".join(label_list))

    return y_list, y_label_list


def decode(logits, lengths, matrix, config):
    """
    :param logits: [batch_size, num_steps, num_tags]float32, logits
    :param lengths: [batch_size]int32, real length of each sequence
    :param matrix: transaction matrix for inference
    :return:
    """
    # inference final labels usa viterbi Algorithm
    paths = []
    small = -1000.0
    start = np.asarray([[small] * config.relation_num[config.dataset] + [0]])
    # print('length:', lengths)
    for score, length in zip(logits, lengths):
        score = score[:length]
        pad = small * np.ones([length, 1])
        logits = np.concatenate([score, pad], axis=1)
        logits = np.concatenate([start, logits], axis=0)
        path, _ = viterbi_decode(logits, matrix)

        paths.append(path[1:])

    return paths


def set_operation(row):
    content_list = row.split(';')
    content_list_after_set = list(set(content_list))
    return ";".join(content_list_after_set)


def get_P_R_F(dev_pd):
    dev_pd = dev_pd.fillna("0")
    dev_pd['y_pred_label'] = dev_pd['y_pred_label'].apply(set_operation)
    dev_pd['y_true_label'] = dev_pd['y_true_label'].apply(set_operation)
    y_true_label_list = list(dev_pd['y_true_label'])
    y_pred_label_list = list(dev_pd['y_pred_label'])
    # print(y_pred_label_list)
    TP = 0
    FP = 0
    FN = 0
    for i, y_true_label in enumerate(y_true_label_list):
        y_pred_label = y_pred_label_list[i].split(';')
        y_true_label = y_true_label.split(';')
        current_TP = 0
        for y_pred in y_pred_label:
            if y_pred in y_true_label:
                current_TP += 1
            else:
                FP += 1
        TP += current_TP
        FN += (len(y_true_label) - current_TP)

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    try:
        F = 2 * P * R / (P + R)
    except:
        F = 0
    return P, R, F


def set_test(model, test_iter, session, out_dir, epoch, config):
    if not test_iter.is_test:
        test_iter.is_test = True

    y_pred_list = []
    y_true_list = []
    ldct_list_tokens = []
    cum_step=0
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length, tokens_list, radical_ids_list, radical_lengths_list in tqdm.tqdm(
            test_iter):

        feed_dict = {
            model.input_x_word: input_ids_list,
            model.input_x_len: seq_length,
            model.input_relation: label_ids_list,
            model.input_mask: input_mask_list,

            model.keep_prob: 1,
            model.is_training: False,
        }
        loss,lengths, logits, trans = session.run(
            fetches=[model.loss, model.lengths, model.logits, model.trans],
            feed_dict=feed_dict
        )


        if cum_step % 10 == 0:
            format_str = 'loss {:.4f}'
            logger.info(
                format_str.format(loss)
            )
        cum_step += 1

        predict = decode(logits, lengths, trans, config)
        y_pred_list.append(predict)
        y_true_list.append(label_ids_list)
        ldct_list_tokens.append(tokens_list)

    ldct_list_tokens = np.concatenate(ldct_list_tokens)
    ldct_list_text = []
    for tokens in ldct_list_tokens:
        text = "".join(tokens)
        ldct_list_text.append(text)

    # 获取验证集文本及其标签
    y_pred_list, y_pred_label_list = get_text_and_label(ldct_list_tokens, y_pred_list)
    y_true_list, y_true_label_list = get_text_and_label(ldct_list_tokens, y_true_list)



    if not os.path.exists(os.path.join(out_dir, 'result')):
        os.makedirs(os.path.join(out_dir, 'result'))
        # os.mkdirs(os.path.join(out_dir, 'result'))

    lable_path = os.path.join(out_dir, 'result', 'label_' + str(epoch) + '.txt')
    lines = []

    with open(lable_path, 'w', encoding='utf-8') as f:
        for i, items in enumerate(y_pred_list):
            text_seq = ldct_list_text[i]
            text_seq = text_seq.replace('[CLS]', '*')
            for j in range(len(items))[1:-1]:
                if y_pred_list[i][j] > len(config.labels):  # 出现非法标签，默认归为‘0’
                    print('出现非法标签：', y_pred_list[i][j])
                    lines.append(
                        "{} {} {}\n".format(text_seq[j], config.labels[y_true_list[i][j] - 1], config.labels[0]))
                else:
                    lines.append("{} {} {}\n".format(text_seq[j], config.labels[y_true_list[i][j] - 1],
                                                     config.labels[y_pred_list[i][j] - 1]))
            lines.append('\n')
        f.writelines(lines)
    eval_perl = "./conlleval_rev.pl"
    metric_path = os.path.join(out_dir, 'result', 'result_metric_' + str(epoch))
    os.system("C:\\Perl64\\bin\\perl.exe {} < {} > {}".format(eval_perl, lable_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]

    for metric in metrics:
        print(metric)
    data = metrics[1].strip().split(';')
    accuracy = float(data[0].split(':')[1].replace('%', ''))
    precision = float(data[1].split(':')[1].replace('%', ''))
    recall = float(data[2].split(':')[1].replace('%', ''))
    F = float(data[3].split(':')[1])
    return accuracy, precision, recall, F



if __name__ == '__main__':

    config = Config()
    learning_rate = 0.001
    config.use_test = True
    config.batch_size = 64
    models = ['bilstm']
    alpha = 0.1
    dataset = 'ccks2017'
    for model in models:

        config.model_name = 'BERT-' + model + '-AT-0.1-0.001-validation'
        print(config.model_name)
        # 模型参数设置
        config.batch_size = 64
        config.dropout = 0.75
        # 下接结构的学习率
        config.learning_rate = learning_rate

        # 是否采用原始BERT
        config.use_origin_bert = True
        # 是否微调
        config.fine_tuning = False
        config.is_training = True
        config.checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
        # 是否采用微调的模型
        config.use_fine_tuned = False
        if config.use_fine_tuned:
            config.checkpoint_path = config.fine_tuned_path[dataset]
        # 训练迭代次数
        config.train_epoch = 50
        # 设置训练的数据集 msra,resume,AgCNER,ccks2017,ribao,clue
        config.dataset = dataset
        config.use_layer_norm = True

        # 模型的类型 使用idcnn,bilstm,crf_only
        config.model_type = model

        # config.use_attention = False
        # 是否采用对抗训练
        config.adversarial = False
        # 设置alpha参数
        config.alpha = alpha

        timestamp = str(int(time()))
        out_dir = os.path.abspath(
            os.path.join(config.model_dir, config.dataset, timestamp))
        timestamp = str(int(time()))
        out_dir = os.path.abspath(
            os.path.join(config.model_dir, config.dataset, timestamp))
        config.out_dir = out_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(out_dir + "/model/"):
            os.makedirs(out_dir + "/model/")
        with open(out_dir + '/' + 'config.json', 'w', encoding='utf-8') as file:
            json.dump(config.__dict__, file)
        print("Writing to {}\n".format(out_dir))
        logger = get_logger(os.path.join(out_dir, 'log.txt'))

        result_data_dir = os.path.join(config.new_data_process_quarter_final, config.dataset)

        config.labels = config.tag2label_mapping[config.dataset]
        vocab_file = config.vocab_file  # 通用词典

        do_lower_case = False
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


        train_iter = DataIterator(config.batch_size, data_file=result_data_dir + '/train.txt', use_bert=config.use_bert,
                                  tokenizer=tokenizer, config=config, gen_path=result_data_dir + '/train.pkl',
                                  seq_length=config.sequence_length,
                                  )
        logger.info('GET!!')
        dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + '/test.txt', config=config,
                                gen_path=result_data_dir + '/dev.pkl', use_bert=config.use_bert, tokenizer=tokenizer,
                                seq_length=config.sequence_length, is_test=True)
        test_iter = DataIterator(config.batch_size, data_file=result_data_dir + '/dev.txt', config=config,
                                 gen_path=result_data_dir + '/test.pkl', use_bert=config.use_bert, tokenizer=tokenizer,
                                 seq_length=config.sequence_length, is_test=True)
        train(train_iter, dev_iter, test_iter, config)


