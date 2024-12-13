# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import random
import re
import time
import jieba
from utils import Vocabulary

random.seed(time.time())


class DataManager:
    def __init__(self):
        if not os.path.exists('cache'):
            os.makedirs('cache')
        
        # 添加成语到Jieba字典
        with open('idiomList_converted.txt', 'r', encoding='utf-8') as f:
            for idiom in f:
                jieba.add_word(idiom.strip())
        
        # 添加 `#idiom#` 到Jieba字典
        jieba.add_word("#idiom#")
        
        if os.path.exists("cache/vocab.pkl"):
            self.vocab = pickle.load(open("cache/vocab.pkl", "rb"))
        else:
            self.vocab = Vocabulary()
            pickle.dump(self.vocab, open("cache/vocab.pkl", "wb"), protocol=2)
        
        print("*** Finish building vocabulary")


    def get_num(self):
        num_word, num_idiom = len(self.vocab.id2word) - 3, len(self.vocab.id2idiom) - 1
        print("Numbers of words and idioms: %d %d" % (num_word, num_idiom))
        return num_word, num_idiom


    def _prepare_data(self, temp_data):
        truth = temp_data["groundTruth"]
        cans = temp_data["candidates"]
        labs = []
        for can, idiom in zip(cans, truth):
            labs.append(can.index(idiom))
        cans = [[self.vocab.tran2id(each, True) for each in each_cans] for each_cans in cans]

        content = temp_data["content"]
        tokens = list(jieba.cut(content))  # 对内容进行分词
        print(f"Tokens: {tokens}")  # 添加调试信息
        doc = []
        loc = []

        for i, token in enumerate(tokens):
            doc.append(self.vocab.tran2id(token))
            if token == "#idiom#":
                loc.append(i)

        # 添加调试信息
        if len(loc) != len(truth):
            print(f"AssertionError: len(loc)={len(loc)}, len(truth)={len(truth)}")
            print(f"Content: {content}")
            print(f"GroundTruth: {truth}")
            print(f"Locations: {loc}")
            # raise AssertionError("Mismatch between `#idiom#` in content and `groundTruth`")
            print(f"Skipping invalid data: {content}, groundTruth: {truth}")
            return None, None, None, None

        return doc, cans, labs, loc


    # def train(self):
    #     file = open("./data/train_data_sampled.txt")
    #     lines = file.readlines()
    #     random.shuffle(lines)
    #     for line in lines:
    #         temp_data = eval(line)
    #         doc, cans, labs, loc = self._prepare_data(temp_data)
    #         del temp_data
    #         yield doc, cans, labs, loc

    def train(self):
        file = open("./data/train_data_sampled.txt", 'r', encoding='utf-8')
        lines = file.readlines()
        random.shuffle(lines)
        total_samples = len(lines)
        valid_samples = 0
        skipped_samples = 0
        for line in lines:
            temp_data = eval(line)
            try:
                doc, cans, labs, loc = self._prepare_data(temp_data)
                valid_samples += 1
                yield doc, cans, labs, loc
            except AssertionError:
                skipped_samples += 1
                continue
        print(f"Total samples: {total_samples}, Valid samples: {valid_samples}, Skipped samples: {skipped_samples}")


    def valid(self, mode): # "dev" or "test" or "out"
        if mode == "dev":
            file = open("./data/dev_data_sampled.txt")
        elif mode == "test":
            file = open("./data/test_data_sampled.txt", 'r', encoding='utf-8')
        else:
            raise EOFError

        for line in file:
            temp_data = eval(line)
            doc, cans, labs, loc = self._prepare_data(temp_data)
            yield doc, cans, labs, loc


    def get_embed_matrix(self):  # DataManager
        if os.path.exists("cache/word_embed_matrix.npy") and os.path.exists("cache/idiom_embed_matrix.npy"):
            self.word_embed_matrix = np.load("cache/word_embed_matrix.npy")
            self.idiom_embed_matrix = np.load("cache/idiom_embed_matrix.npy")

        else:
            np.random.seed(37)
            def embed_matrix(file, dic, dim=200):
                fr = open(file, encoding="utf8")
                wv = {}
                for line in fr:
                    vec = line.split(" ")
                    word = vec[0]
                    if word in dic:
                        vec = [float(value) for value in vec[1:]]
                        assert len(vec) == dim
                        wv[dic[word]] = vec
                        # which indicates the order filling in wv is the same as id2idiom/id2word

                lost_cnt = 0
                matrix = []
                for i in range(len(dic)):
                    if i in wv:
                        matrix.append(wv[i])
                    else:
                        lost_cnt += 1
                        matrix.append(np.random.uniform(-0.1, 0.1, [dim]))
                
                # 添加 UNK 嵌入向量
                matrix.append(np.random.uniform(-0.1, 0.1, [dim]))
                return matrix, lost_cnt

            self.word_embed_matrix, lost_word = embed_matrix("../data/wordvector.txt", self.vocab.word2id)
            self.idiom_embed_matrix, lost_idiom = embed_matrix("../data/idiomvector.txt", self.vocab.idiom2id)

            self.word_embed_matrix = np.array(self.word_embed_matrix, dtype=np.float32)
            self.idiom_embed_matrix = np.array(self.idiom_embed_matrix, dtype=np.float32)
            np.save("cache/word_embed_matrix.npy", self.word_embed_matrix)
            np.save("cache/idiom_embed_matrix.npy", self.idiom_embed_matrix)
            print("*** %d idioms and %d words not found" % (lost_idiom, lost_word))

        print("*** Embed matrixs built")
        return self.word_embed_matrix, self.idiom_embed_matrix
