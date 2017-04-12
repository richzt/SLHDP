#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import jieba

out_dir = os.path.abspath(os.path.curdir)    # 当前目录
stopwords = [line.strip() for line in open(out_dir + '/data/stopwords.txt').readlines()]    # 停用词集


def load_file(filename, excluds_stopwords, df):
    final_set = []              # 文档集
    label_set = []              # 标签集
    dircout = 0                 # 类别计数
    walk = os.walk(filename)    # 数据集,'E:\python\code\20_newsgroups'
    for root, dirs, files in walk:
        if not files:
            continue
        print(str(dircout)+'类文档数据分词处理中...')
        for name in files:                             # 遍历所有文件，包括子文件夹中的。此处为1篇文章
            f = open(os.path.join(root, name), 'r')
            raw = f.read()
            f.close()
            raw = raw.replace("\n", "").replace(" ", "")
            try:
                word_list = jieba.cut(raw, cut_all=False)    # 英文分句,1文章-N个(短句+词)
            except Exception as e:
                print(e, root, dirs, name)
                continue
            voca = Vocabulary(excluds_stopwords, df)   # 将这个类赋给voca
            doc = voca.doc_deal(word_list)             # 文档集的word_id
            if len(doc) == 0:
                continue
            label_set.append(dircout)                  # 标签
            final_set.append(" ".join(doc))            # 全部文章
        dircout += 1
    return np.array(final_set), np.array(label_set)


class Vocabulary:                                      # 去低频词
    def __init__(self, excluds_stopwords=None, df=0):
        self.excluds_stopwords = excluds_stopwords
        self.df = df                                   # 词频

    def check_chinese(self, check_str):                # unicode码，仅限汉字
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

    def doc_deal(self, doc):
        doc_temp = []
        for term in doc:                                    # 每一个词
            if term in self.excluds_stopwords or (not self.check_chinese(term)):
                continue
            if term.encode('utf-8') not in stopwords and (not isinstance(term, float)):       # 去停用词
                doc_temp.append(unicode(term))      # 读取格式如果不是utf-8，存储时出现utf8编码不识别错误，用Unicode()
        return self.cut_low_freq(doc_temp)

    def cut_low_freq(self, doc):
        words_df = set(word for word in set(doc) if doc.count(word) <= self.df)    # 提取低频词集合
        doc = [stem for stem in doc if stem not in words_df]
        return doc


def saveCutExcel(data, Path):                               # 保存每篇文本
    if os.path.exists(Path):
        os.remove(Path)
    data.to_excel(Path, sheet_name='Sheet1', index=None)    # 不要索引列


def textDeal(filename, excluds_stopwords, df, percentage):
    print('文件夹数据读取中...')
    final_set, label_set = load_file(filename, excluds_stopwords, df)
    shuffle_indices = np.random.permutation(np.arange(len(label_set)))             # 随机打乱索引
    x_shuffled = final_set[shuffle_indices]
    y_shuffled = label_set[shuffle_indices]

    pre, now = int((percentage[0]/10.)*len(y_shuffled)), int(((10-percentage[2])/10.)*len(y_shuffled))
    x_para, x_train, x_test = x_shuffled[:pre], x_shuffled[pre:now], x_shuffled[now:]
    y_para, y_train, y_test = y_shuffled[:pre], y_shuffled[pre:now], y_shuffled[now:]

    corpus_para = pd.DataFrame({'text': x_para, 'label': y_para})
    corpus_train = pd.DataFrame({'text': x_train, 'label': y_train})
    corpus_test = pd.DataFrame({'text': x_test, 'label': y_test})
    corpus_para.reset_index(drop=True, inplace=True)
    corpus_train.reset_index(drop=True, inplace=True)
    corpus_test.reset_index(drop=True, inplace=True)
    saveCutExcel(corpus_para, './data/corpus_para.xlsx')
    saveCutExcel(corpus_train, './data/corpus_train.xlsx')
    saveCutExcel(corpus_test, './data/corpus_test.xlsx')
    print('预处理完成！')

if __name__ == "__main__":
    filename = out_dir + '\data\Sogouexp'                    # 文件或文件路径
    excluds_stopwords = []                                  # 额外的停用词
    df = 1                                                  # 去掉<=df的低频词
    percentage = [2, 5, 3]                                  # 参数集、训练集、测试集占比
    textDeal(filename, excluds_stopwords, df, percentage)   # 预处理文本
