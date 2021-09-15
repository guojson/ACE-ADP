class Config:

    def __init__(self):
        self.model_name = ''
        self.embed_dense = True
        self.embed_dense_dim = 512  # 对BERT的Embedding降维
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.keep_prob = 0.75

        self.use_test=False
        self.use_caps = False
        self.iter_routing = 3
        self.stddev = 0.01
        self.img_dim = 64

        self.decay_rate = 0.9
        self.decay_step = 1000
        self.num_checkpoints = 20 * 3

        self.train_epoch = 30
        self.sequence_length = 300  # BERT的输入MAX_LEN

        self.learning_rate = 1e-4  # 下接结构的学习率
        self.embed_learning_rate = 5e-5  # BERT的微调学习率
        self.fine_tuning = False  # 是否微调
        self.is_training = False
        self.batch_size = 32

        self.out_dir = ''
        # BERT预训练模型的存放地址
        self.bert_file = './chinese_L-12_H-768_A-12/bert_model.ckpt'
        self.bert_config_file = './chinese_L-12_H-768_A-12/bert_config.json'
        self.vocab_file = './chinese_L-12_H-768_A-12/vocab.txt'

        # predict.py ensemble.py get_ensemble_final_result.py post_ensemble_final_result.py的结果路径
        self.continue_training = False
        self.ensemble_source_file = './ensemble/source_file/'
        self.ensemble_result_file = './ensemble/result_file/'

        # 是否采用微调的模型
        self.use_fine_tuned = False
        # 存放的模型名称
        self.checkpoint_path = "./model/runs_7/1577502293/model_0.5630_0.6378-10305"  #

        self.model_dir = './model/'  # 模型存放地址
        self.new_data_process_quarter_final = './data/'  # 数据预处理的结果路径
        self.source_data_dir = './data'  # 原始数据集

        self.use_attention = False
        self.attention_unit = 300
        self.use_layer_norm = False
        self.model_type = 'bilstm'  # 使用idcnn,bilstm
        self.lstm_dim = 256
        self.dropout = 0.5
        self.use_origin_bert = True  # True:使用原生bert, False:使用动态融合bert

        self.dataset = 'ccks2017'

        # 是否适用对抗
        self.adversarial = False
        self.alpha = 0.01  # CNN 0.05， RNN 0.1

        tag2label_CCKS2017_ner = [
            '0', 'B-body', 'I-body', 'B-symp', 'I-symp', 'B-dise', 'I-dise', 'B-chec', 'I-chec', 'B-cure', 'I-cure']
        tag2label_AgCNER_ner = ['0',
                                'B-CRO', 'I-CRO', 'B-DIS', 'I-DIS', 'B-PET', 'I-PET', 'B-DRUG', 'I-DRUG', 'B-FER',
                                'I-FER', 'B-REA', 'I-REA',
                                'B-WEE', 'I-WEE', 'B-CLA', 'I-CLA', 'B-PER', 'I-PER', 'B-PART', 'I-PART', 'B-STRAINS',
                                'I-STRAINS', 'B-SYM', 'I-SYM']
        tag2label_resume_ner = ['0', 'B-NAME', 'I-NAME', 'B-CONT', 'I-CONT', 'B-RACE', 'I-RACE', 'B-TITLE', 'I-TITLE',
                                'B-EDU', 'I-EDU', 'B-ORG', 'I-ORG', 'B-PRO', 'I-PRO', 'B-LOC', 'I-LOC']
        tag2label_cluner_ner = ['0', 'B-company', 'I-company', 'B-name', 'I-name', 'B-game', 'I-game', 'B-organization',
                                'I-organization',
                                'B-movie', 'I-movie', 'B-position', 'I-position', 'B-address', 'I-address',
                                'B-government', 'I-government',
                                'B-scene', 'I-scene', 'B-book', 'I-book']
        self.tag2label_mapping = {
            'ccks2017': tag2label_CCKS2017_ner,
            'AgCNER': tag2label_AgCNER_ner,
            'resume': tag2label_resume_ner,
            'clue': tag2label_cluner_ner
        }
        self.relation_num = {
            'ccks2017': 14,
            'AgCNER': 28,
            'resume': 20,
            'clue': 24
        }
        self.fine_tuned_path = {
            'ccks2017': r'F:\data\CCF_ner\model\ccks2017\bilstm-fine-tuning\model\model_86.06_91.35_88.62_0.00-251',
            'AgCNER': r'E:\0-科研学习\实验结果\AgCNER\BERT-BiLSTM-CRF微调模型\model\model_93.56_94.79_94.17_4.00-12505',
            'msra': r'F:\data\CCF_ner\model\msra\bilstm-fine-tuning\model\model_89.09_91.71_90.38_0.00-5625',
            'resume': r'J:\paper7\CCF_ner\model\resume\BERT-BiLSTM-CRF-0.0002-FT-第二次\model\model_90.60_95.77_93.11_0.00-383',
            'clue': r'F:\data\CCF_ner\model\clue\bilstm-fine-tuning\model\model_68.03_73.64_70.72_0.00-1344'
        }

        self.labels = []
