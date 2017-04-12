#!/Usr/bin/python
# -*- coding: utf-8 -*-
import time, os
from gensim import corpora
import pandas as pd
import SLHDP, HDPgibbs, HDPbeta, SLLDA
import pre_deal
import numpy as np

out_dir = os.path.abspath(os.path.curdir)    # 当前目录


class schedule():
    def __init__(self, predealN, iteration, corpus_cont):
        self.iteration = iteration    # 迭代次数,已抛弃前n次迭代的结果
        self.predeal = predealN
        self.acc_style = 0            # 文档分类标准
        self.corpus_cont = corpus_cont

        self.percentage = [2, 5, 3]   # 参数集、训练集、测试集占比
        if self.predeal == 0:
            filename = out_dir + '\data\1000+'                # 文件或文件路径
            excluds_stopwords = []                                   # 额外的停用词
            df = 1                                                   # 去掉<=df的低频词
            preDeal.textDeal(filename, excluds_stopwords, df, self.percentage)      # 预处理文本

    def schedule_core(self, parameter, model_style):
        corpus_name = {0: 'para', 1: 'train', 2: 'test'}
        dataPath = out_dir + '\data\corpus_'+corpus_name[self.corpus_cont]+'.xlsx'
        data = pd.read_excel(dataPath, 'Sheet1', index_col=None, encoding='gbk')
        texts = []
        for line in data['text']:
            try:
                texts.append(line.split(" "))
            except Exception as e:
                print e,line
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(1, 0.2)                     # 过滤出现低于几篇文档的词和高于0.2*num_docs的词
        V = len(dictionary)
        corpus = [dictionary.doc2bow(text) for text in texts]  # 语料利用doc2bow转化为向量(元组)的形式

        text_dic = dictionary.token2id  # 字典，{u'minors': 29, u'generation': 21,...
        text_dic2 = sorted(text_dic.iteritems(), key=lambda d: d[1], reverse=False)   # 获取按id号排序的词典
        voca = [dic2[0] for dic2 in text_dic2]
        if 0 in model_style:
            [alpha, gamma, base] = parameter[0]
            print("data=%d Vwords=%d alpha=%.3f gamma=%.3f base=%.3f" % (len(corpus), V, alpha, gamma, base))
            t1 = time.time()
            hdp = HDPgibbs.HDP(alpha, gamma, base, V, corpus)     # initial
            HDPgibbs.hdp_learning(hdp, self.iteration, voca, parameter, corpus, data['label'], self.acc_style)
            print 'HDPgibbs运行时间：' + str((time.time() - t1))
        if 1 in model_style:
            [alpha, gamma, base] = parameter[1]
            print("data=%d Vwords=%d alpha=%.3f gamma=%.3f base=%.3f" % (len(corpus), V, alpha, gamma, base))
            t1 = time.time()
            hdp = HDPbeta.HDP(alpha, gamma, base, V, corpus)       # initial
            pfile = HDPbeta.hdp_learning(hdp, self.iteration, voca, parameter, corpus, data['label'], self.acc_style)
            pfile.writelines('HDPbeta运行时间：' + str((time.time() - t1))+ '\n')
            pfile.close()
        if 2 in model_style:
            [alpha, gamma, base, beta_xin, labels] = parameter[2]
            print "data=%d Vwords=%d alpha=%.3f gamma=%.3f base=%.3f beta_xin=%.2f labels=%d" \
                  % (len(corpus), V, alpha, gamma, base, beta_xin, labels)
            t1 = time.time()
            slhdp = SLHDP.SLHDP(alpha, gamma, base, beta_xin, V, labels, corpus)    # initial
            pfile = SLHDP.slhdp_learning(slhdp, self.iteration, voca, parameter, corpus, data['label'], self.acc_style)
            pfile.writelines('SLHDP运行时间：' + str((time.time() - t1))+ '\n')
            pfile.close()

# para_a = [0.7, 1.3]   #alpha范围         
# para_y = [0.7, 1.3]   #gamma范围         
# para_a = np.random.gamma(0.1, 0.1)   # `alpha`: second level concentration
# para_y = np.random.gamma(5, 0.1)     # `gamma`: first level concentration
# para_b = [0.1, 0.3, 0.5, 0.7, 0.9]   #base（lamda）范围 `eta`: the topic Dirichlet
# para_bx= [0.5, 0.6, 0.7, 0.8, 0.9]   #beta_xin范围

if __name__ == "__main__":

    # np.random.seed(0)
    predeal = 1                           # 0文本预处理
    iteration = 1000                      # 迭代次数
    model_style = [2]                     # 0:hdpg,1:hdpb,2:slhdp,3:sllda
    corpus_cont = 1                       # 0:参数集,1:训练集,2:测试集
    hdpgibbs = [0.01, 0.5, 0.5]           # alpha, gamma, base
    hdpbeta = [0.01, 0.5, 0.5]            # alpha, gamma, base
    slhdp = [0.01, 0.5, 0.5, 0.7, 6]      # alpha, gamma, base, beta_xin, labels
    sllda = [0.01, 0.5, 0.7, 9, 3]        # alpha, base, beta_xin, labels, h_label
    parameters = [[hdpgibbs, hdpbeta, slhdp, sllda]]

    schedule = schedule(predeal, iteration, corpus_cont)
    for parameter in parameters:
        schedule.schedule_core(parameter, model_style)
