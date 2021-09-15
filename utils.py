import logging
import pickle
import tensorflow as tf
import numpy as np

from bert import tokenization
from tqdm import tqdm
from config import Config
import os as os
from time import *


def load_data(data_file):
    data = []
    with open(data_file, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        # print(line)
        if line != '\n':
            if len(line.strip()) > 1:
                [char, label] = line.strip().split()
                if label.strip() == 'O':
                    label = '0'
                sent_.append(char.strip())
                tag_.append(label)
        else:
            if len(sent_) > 0 and len(tag_) > 0:
                data.append([' '.join(tag_), ' '.join(sent_)])
            sent_, tag_ = [], []

    return data


def create_example(lines):
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s" % i
        text = tokenization.convert_to_unicode(line[1])
        label = tokenization.convert_to_unicode(line[0])
        examples.append(InputExample(guid=guid, text=text, label=label))
    return examples


def get_examples(data_file):
    return create_example(
        load_data(data_file)
    )


def get_labels(labels):
    # config=Config()
    # # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]", '']
    # labels=config.tag2label_mapping[config.dataset]
    if 'X' not in labels:
        labels.append('X')
    if '[CLS]' not in labels:
        labels.append('[CLS]')
    if '[SEP]' not in labels:
        labels.append('[SEP]')
    # labels.append('[CLS]')
    # labels.append('[SEP]')
    # labels.append('')
    return labels

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class DataIterator:
    """
    数据迭代器
    """

    def __init__(self, batch_size, data_file, tokenizer, config, gen_path=None, use_bert=False, seq_length=100,
                 is_test=False):
        self.data_file = data_file
        self.data = get_examples(data_file)
        self.batch_size = batch_size
        self.use_bert = use_bert
        self.seq_length = seq_length
        self.num_records = len(self.data)
        self.all_tags = []
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test
        if not self.is_test:
            self.shuffle()
        self.tokenizer = tokenizer
        self.label_map = {}
        for (i, label) in enumerate(get_labels(config.labels), 1):
            self.label_map[label] = i
        print(self.label_map)
        self.unknow_tokens = self.get_unk_token()

        print(self.unknow_tokens)
        print(self.num_records)

        # self.pro_data={}
        # if not os.path.exists(gen_path):
        #     self.generate_records(gen_path)
        # else:
        #     self.read_records(gen_path)

    def get_unk_token(self):
        unknow_token = set()
        for example_idx in self.all_idx:
            textlist = self.data[example_idx].text.split(' ')

            for i, word in enumerate(textlist):
                token = self.tokenizer.tokenize(word)

                if '[UNK]' in token:
                    unknow_token.add(word)
        return unknow_token

    def convert_single_example(self, example_idx):
        textlist = self.data[example_idx].text.split(' ')
        labellist = self.data[example_idx].label.split(' ')
        tokens = textlist  # 区分大小写
        labels = labellist

        if len(tokens) >= self.seq_length - 1:
            tokens = tokens[0:(self.seq_length - 2)]
            labels = labels[0:(self.seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(self.label_map["[CLS]"])

        upper_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                        'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                        ]
        # print(tokens)
        # print(labels)
        for i, token in enumerate(tokens):
            if token in self.unknow_tokens and token not in upper_letter:
                token = '[UNK]'
                ntokens.append(token)  # 全部转换成小写, 方便BERT词典
            else:
                ntokens.append(token.lower())  # 全部转换成小写, 方便BERT词典
            segment_ids.append(0)

            label_ids.append(self.label_map[labels[i]])

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        ntokens.append("[SEP]")

        segment_ids.append(0)
        label_ids.append(self.label_map["[SEP]"])

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            ntokens.append("**NULL**")
            tokens.append("**NULL**")



        assert len(input_ids) == self.seq_length
        assert len(input_mask) == self.seq_length
        assert len(segment_ids) == self.seq_length
        assert len(label_ids) == self.seq_length
        assert len(tokens) == self.seq_length
        return input_ids, input_mask, segment_ids, label_ids, tokens

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    # def generate_records(self,path):
    #     print('the pickle process is runing...')
    #     for index in tqdm(range(self.num_records)):
    #         input_ids, input_mask, segment_ids, label_ids, tokens=self.convert_single_example(index)
    #         self.pro_data[index]=[input_ids, input_mask, segment_ids, label_ids, tokens]
    #     with open(path,'wb') as fw:
    #         pickle.dump(self.pro_data,fw)
    # def read_records(self,path):
    #     print('file is existing, reading process is runing')
    #     path=os.path.join(path)
    #     with open(path,'rb') as fr:
    #         self.pro_data=pickle.load(fr)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件
            self.idx = 0
            if self.is_test == False:
                self.shuffle()
            raise StopIteration

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        label_ids_list = []
        tokens_list = []
        radical_ids_list = []
        radical_lengths_list = []

        num_tags = 0
        while num_tags < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)
            # res=self.pro_data[idx]
            if res is None:
                self.idx += 1
                if self.idx >= self.num_records:
                    break
                continue
            input_ids, input_mask, segment_ids, label_ids, tokens = res

            # 一个batch的输入
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            label_ids_list.append(label_ids)
            tokens_list.append(tokens)
            if self.use_bert:
                num_tags += 1

            self.idx += 1
            if self.idx >= self.num_records:
                break

        return input_ids_list, input_mask_list, segment_ids_list, label_ids_list, self.seq_length, tokens_list, radical_ids_list, radical_lengths_list


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data   [([a,a,a,a,a,a],[b,b,b,b,b,b]),([a,a,a,a,a,a],[b,b,b,b,b,b])]
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            if len(line.strip().split()) == 2:
                [char, label] = line.strip().split()
                if label == 'O':
                    label = '0'
                sent_.append(char)
                tag_.append(label)
        else:
            if len(sent_) > 0 and len(tag_) > 0:
                data.append((sent_, tag_))
            sent_, tag_ = [], []
    print(len(data))
    return data


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return (shape)
