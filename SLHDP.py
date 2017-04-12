#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import mode
import os, operator
from six import iteritems
out_dir = os.path.abspath(os.path.curdir)    # 当前目录


class SLHDP:
    def __init__(self, alpha, gamma, base, beta_xin, V, labels, docs):  # terms,顾客
        self.alpha = float(alpha)        # a G_j
        self.gamma = float(gamma)        # y G_0
        self.base = float(base)          # lamda  H
        self.beta_xin = float(beta_xin)  # 约束参数
        self.V = V
        self.k_size = labels

        self.beta = np.zeros(self.k_size,dtype=float)
        for k in xrange(self.k_size):
            self.beta[k] = self.beta_xin / self.k_size
        # self.beta = [self.beta_xin / self.k_size] * self.k_size
        self.beta_u = 1 - self.beta_xin

        self.x_ji = docs
        self.t_ji = [np.zeros(len(x_i), dtype=int) - 1 for x_i in docs]
        # J*N table for each document and term (without assignment)
        self.k_jt = [[] for _ in docs]                                # J topic for each document and table
        self.k_ji = [np.zeros(len(x_i), dtype=int) for x_i in docs]     # 单词对应的k
        self.d_k = [{} for _ in docs]
        self.d_maxk = []
        self.n_jt = [np.ndarray(0, dtype=int) for _ in docs]  # J number of terms for each document and table

        self.tables = [[] for _ in docs]                    # J available id of tables for each document
        self.n_tables = 0
        self.topics = range(labels)                           # available id of topics

        self.m_k = np.zeros(self.k_size,dtype=int)       # [] number of tables for each topic
        self.n_k = np.zeros(self.k_size,dtype=int)       # [] number of terms for each topic
        self.n_kv = np.zeros((self.k_size, V),dtype=int)  # [] number of terms for each topic and vocabulary

        # memoization
        self.updated_n_beta()
        self.Vbase = V * base
        self.beta_u_f_k_new_x_ji = self.beta_u / V
        self.cur_log_base_cache = [0]
        self.cur_log_V_base_cache = [0]

    def inference(self):                       # sample t & k
        for j, x_i in enumerate(self.x_ji):    # 文档subscript，文档内容id码
            for i in xrange(len(x_i)):         # 该文档中的词subscript
                self.sampling_table(j, i)
            for t in self.tables[j]:
                # if ((k_old < self.k_size) and self.m_k[k_old] > 1) or (k_old >= self.k_size):#控制初始t少时，topics.remove问题
                self.sampling_k(j, t)
                # else: continue
        self.sampling_beta()

        for j ,t_i in enumerate(self.t_ji):
            for i in xrange(len(t_i)):         # 生成k_ji矩阵
                t_ji = self.t_ji[j][i]
                k_jt = self.k_jt[j][t_ji]
                self.k_ji[j][i] = k_jt

    def worddist(self):
        return [(self.n_kv[k] + self.base) / (self.n_k[k] + self.Vbase) for k in self.topics]
    def worddist1(self):
        phi=dict()
        for k in self.topics:phi[k]=(self.n_kv[k] + self.base) / (self.n_k[k] + self.Vbase)
        return phi

    def perplexity(self):
        phi = self.worddist()
        phi.append(np.zeros(self.V) + 1.0 / self.V)
        log_per = 0
        N = 0
        gamma_over_T_gamma = self.gamma / (self.n_tables + self.gamma)
        for j, x_i in enumerate(self.x_ji):
            p_k = np.zeros(self.m_k.size)       # topic dist for document
            for t in self.tables[j]:
                k = self.k_jt[j][t]
                p_k[k] += self.n_jt[j][t]       # n_jk,第j文档下各k的计数
            len_x_alpha = len(x_i) + self.alpha
            p_k /= len_x_alpha
            p_k_parent = self.alpha / len_x_alpha
            p_k += p_k_parent * (self.m_k / (self.n_tables + self.gamma))

            theta = [p_k[k] for k in self.topics]
            theta.append(p_k_parent * gamma_over_T_gamma)
            for v in x_i:
                if np.inner([p[v[0]] for p in phi], theta) != 0:
                    log_per -= np.log(np.inner([p[v[0]] for p in phi], theta))
                else:
                    log_per = np.exp(np.power(N, N))
            N += len(x_i)
        return np.exp(log_per / N)

    def dump(self, disp_x=False):
        if disp_x: print "x_ji:", self.x_ji
        print "t_ji:", self.t_ji
        print "k_jt:", self.k_jt
        print "n_kv:", self.n_kv
        print "n_jt:", self.n_jt
        print "n_k:", self.n_k
        print "m_k:", self.m_k
        print "tables:", self.tables
        print "topics:", self.topics

    # internal methods from here

    # cache for faster calcuration
    def updated_n_beta(self):
        self.alpha_over_T_beta = self.alpha /  (np.sum(self.beta) + self.beta_u)

    # sampling t (table) from posterior
    def sampling_table(self, j, i):
        v = self.x_ji[j][i][0]    # 第j篇文档的第i个词id
        tf = self.x_ji[j][i][1]
        t_old = self.t_ji[j][i]   # initial=-1
        tables = self.tables[j]
        if t_old >= 0:            # updata t and k
            k_old = self.k_jt[j][t_old]  # decrease counters
            self.n_kv[k_old, v] -= tf
            self.n_k[k_old] -= tf
            self.n_jt[j][t_old] -= tf

            if self.n_jt[j][t_old] == 0:
                # table that all guests are gone
                tables.remove(t_old)
                self.m_k[k_old] -= 1
                self.n_tables -= 1

                if self.m_k[k_old] == 0:
                    # topic (dish) that all guests are gone
                    # if k_old == 0 or k_old == 1:
                    self.topics.remove(k_old)
                    if k_old >= self.k_size:
                        self.beta_u += self.beta[k_old]
                        self.beta[k_old] = 0
                        self.updated_n_beta()

        # sampling from posterior p(t_ji=t)
        t_new = self.sampling_t(j, i, v, tables)

        # increase counters
        self.t_ji[j][i] = t_new
        self.n_jt[j][t_new] += tf

        k_new = self.k_jt[j][t_new]
        self.n_k[k_new] += tf
        self.n_kv[k_new, v] += tf  # k与单词id维度下的顾客计数

    def sampling_t(self, j, i, v, tables):
        f_k = (self.n_kv[:, v] + self.base) / (self.n_k + self.Vbase)  # fi_kt
        p_t = [self.n_jt[j][t] * f_k[self.k_jt[j][t]] for t in tables]  # p(t)集合，self.k_jt[j][t]]为k值
        p_x_ji = np.inner(self.beta, f_k) + self.beta_u_f_k_new_x_ji
        p_t.append(p_x_ji * self.alpha_over_T_beta) # 附加新table

        p_t /= (np.sum(map(abs, p_t)))  # 绝对值归一化
        drawing = np.random.multinomial(1, p_t).argmax() # argmax() ：searching for satisfy f(X)局部最大的参数subscript
        if drawing < len(tables):
            return tables[drawing]
        else:
            return self.new_table(j, i, f_k)

    # Assign guest x_ji to a new table and draw topic (dish) of the table
    def new_table(self, j, i, f_k):
        # search a spare table ID
        T_j = self.n_jt[j].size  # j中桌子t数
        for t_new in xrange(T_j):      # for(if不出错),则else;否则直接跳过else
            if t_new not in self.tables[j]: break
        else:
            # new table ID (no spare)
            t_new = T_j
            self.n_jt[j].resize(t_new+1)
            self.n_jt[j][t_new] = 0  # last list element assign 0
            self.k_jt[j].append(0)
        self.tables[j].append(t_new)
        self.n_tables += 1

        # sampling of k for new topic(= dish of new table)
        p_k = [self.beta[k] * f_k[k] for k in self.topics]
        p_k.append(self.beta_u_f_k_new_x_ji)
        p_k /= (np.sum(map(abs, p_k)))

        k_new = self.sampling_topic(np.array(p_k, copy=False))    # 给每次概率最大的类付大概率，已达到最快收敛，提升
        self.k_jt[j][t_new] = k_new
        self.m_k[k_new] += 1
        return t_new
    # sampling topic
    # In the case of new topic, allocate resource for parameters

    def sampling_topic(self, p_k):
        drawing = np.random.multinomial(1, p_k).argmax()
        if drawing < len(self.topics):   # 选择已有K
            # existing topic
            k_new = self.topics[drawing]
        else:                            # Knew
            # new topic
            K = self.m_k.size
            for k_new in xrange(K):      # for(if不出错),则else;否则直接跳过else
                # recycle table ID, if a spare ID exists
                if k_new not in self.topics: break  # 循环检验m_k是否与self.topics对应
            else:
                # new table ID, if otherwise
                k_new = K
                self.n_k = np.resize(self.n_k, k_new + 1)
                self.n_k[k_new] = 0
                self.m_k = np.resize(self.m_k, k_new + 1)
                self.m_k[k_new] = 0
                self.beta = np.resize(self.beta, k_new + 1)
                self.beta[k_new] = 0
                self.n_kv = np.resize(self.n_kv, (k_new+1, self.V))
                self.n_kv[k_new, :] = np.zeros(self.V, dtype=int)  # V [[0,0,...,0]]
            self.topics.append(k_new)
            if k_new >= self.k_size:
                self.updata_beta(k_new)
        return k_new

    def updata_beta(self, k_new):
        nu = np.random.beta(self.gamma, 1.0)
        self.beta[k_new] = self.beta_u * nu
        self.beta_u = self.beta_u * (1.0 - nu)

    # sampling b (beta) from posterior
    def sampling_beta(self):
        m_k = []
        m_k.extend(self.m_k.tolist())
        m_k = m_k[self.k_size:]
        m_k.append(self.gamma)
        beta_ku = (1 - self.beta_xin) * np.random.dirichlet(m_k, ).transpose()  # 替代beta_k公式
        if beta_ku.size > 1:
            for k in xrange(len(beta_ku)-1):
                self.beta[k + self.k_size] = beta_ku[k]
        self.beta_u = beta_ku[-1]

    def count_n_jtv(self, j, t, k_old):  # 循环tables中的t
        """count n_jtv and decrease n_kv for k_old"""
        x_i = self.x_ji[j]
        t_i = self.t_ji[j]
        n_jtv = dict()
        tf = 0
        for i, t1 in enumerate(t_i):    # j文档中全部i对应的t
            if t1 == t:                 # 坐在t桌
                v = x_i[i][0]           # 坐在t桌的第i个单词
                tf = x_i[i][1]
                self.n_kv[k_old, v] -= tf
                n_jtv[v] = tf           # t桌对应的所有词{v1:count1,v2:count2,...}
        return n_jtv.items()  # [('v1',7), ('v2',count2)...]

    def log_f_k_new_x_jt(self, n_jt, n_tv, n_kv = None, n_k = 0):
        p = 0
        for (v_l, n_l) in n_tv:
            if not n_kv is None:
                p += np.log(n_kv[v_l] + self.base) - np.log(n_k + self.Vbase)
            else:
                p += np.log(1. / self.V)
        return p

    def sampling_k(self, j, t):
        """sampling k (dish=topic) from posterior"""
        k_old = self.k_jt[j][t]
        n_jt = self.n_jt[j][t]
        self.m_k[k_old] -= 1
        self.n_k[k_old] -= n_jt
        if self.m_k[k_old] == 0:
            self.topics.remove(k_old)
            if k_old >= self.k_size:
                self.beta_u += self.beta[k_old]
                self.beta[k_old] = 0
                self.updated_n_beta()

        # sampling of k
        n_jtv = self.count_n_jtv(j, t, k_old)  # t桌对应的所有词v,{v1:count1,v2:count2,...},sum(count)=n_jt
        K = len(self.topics)
        log_p_k = np.zeros(K+1, dtype=float)
        for i, k in enumerate(self.topics):
            log_p_k[i] = np.log(self.beta[k]) + self.log_f_k_new_x_jt(n_jt, n_jtv, self.n_kv[k, :], self.n_k[k])
        log_p_k[K] = np.log(self.beta_u) + self.log_f_k_new_x_jt(n_jt, n_jtv) #last k

        p_k=np.exp(log_p_k - log_p_k.max())  # far too small,Subtracting the maximum is to suppress overflow/underflow
        # p_k=np.exp(log_p_k)
        p_k /= np.sum(map(abs,p_k))
        k_new = self.sampling_topic(np.array(p_k, copy=False))  # for too small
        # update counters
        self.k_jt[j][t] = k_new
        self.m_k[k_new] += 1
        self.n_k[k_new] += n_jt

        x_i = self.x_ji[j]
        t_i = self.t_ji[j]
        for i, t1 in enumerate(t_i):       # j文档中全部i对应的t
            if t1 == t:                    # 坐在t桌
                v = x_i[i][0]              # 坐在t桌的第i个单词
                tf = x_i[i][1]
                self.n_kv[k_new, v] += tf  # 对应count_n_jtv中的更新

def output_word_topic_dist(slhdp, voca):
    zcount = dict()
    wordcount = dict()
    for each in slhdp.topics:
        zcount[each] = 0
        wordcount[each] = dict()
    for xlist, zlist in zip(slhdp.x_ji, slhdp.k_ji):
        for x, z in zip(xlist, zlist):     # 该词x及其对应的主题z
            zcount[z] += x[1]              # n_k
            x1 = x[0]
            if x1 in wordcount[z]: wordcount[z][x1] += x[1]
            else: wordcount[z][x1] = x[1]
    phi = slhdp.worddist1()
    t_word = [{} for k in xrange(len(slhdp.topics))]
    for k in slhdp.topics:
        flist = []
        str_topic = "-- topic:"+str(k)+ '('+str(zcount[k])+" words)"  # topic words
        phi_k = []
        phi_k.extend((np.argsort(-phi[k])[:10]).tolist())  # n_kv中取出排名前10的词
        for w in phi_k:
            if wordcount[k].get(w,0) != 0:   # 排除计数为0的词
                dic = {}
                dic[voca[w]] = str(np.round(phi[k][w],5))[:10] + '(' + str(wordcount[k].get(w, 0)) + ')'
                flist.append(dic)   # phi权重；#get(w,0)有返回w_value，无返回0
        t_word[slhdp.topics.index(k)][str_topic] = flist
    return t_word


def slhdp_learning(slhdp, iteration, voca, parameter, corpus, label, acc_style):
    pre_perp = slhdp.perplexity()
    print pre_perp
    p=[]
    result_set=[]
    acc_value = []
    # kcc_value = []
    ff = open(out_dir + '/data/slhdp_process.csv', 'w+')
    for i in xrange(iteration+50):
        slhdp.inference()
        if i < 50:   # 不记录前50次的结果
            continue
        new_perp = slhdp.perplexity()
        p.append(new_perp)

        acc = accuracy(slhdp.k_ji, corpus, label, acc_style)
        acc_value.append(acc)
        # kcc_value.append(kcc)
        print "slhdp-%d K=%d p=%f a=%.5f" % (i-50, len(slhdp.topics), new_perp, acc)
        stra = 'slhdp,' + str(i-50) + ',' + str(len(slhdp.topics)) + ',' + str(new_perp) + ',' + str(acc) + '\n'
        ff.writelines(stra)
        t_word=output_word_topic_dist(slhdp, voca)    # 每一次的topic_word结果
        result_set.append(t_word)
    ff.close()
    x = np.argsort(p)     # p值从小到大排序的索引值(下标)

    try:
        s = [int(each) for each in p]
        P_index = str(mode(s)[0][0])
    except Exception:
        print 'please give a number which is less than 1 for base'
        P_index = 0
    m = [len(each) for each in result_set]
    strlist = 'parameter:' + str(parameter) + ',P_index:' + str(P_index) + \
              ',K_index:' + str(mode(m)[0][0]) + ',A_index:' + str(round(np.average(acc_value), 4)) + ']\n'
    pfile = open(out_dir + '/data/process_slhdp_file.txt', 'w+')
    pfile.writelines(strlist)

    pfile.writelines( 'P_index:'+str([round(p[x[0]],8),round(p[x[-1]],8),mode(s)[0][0],mode(s)[1][0]])+'\n')
    # p最小、最大、众数（int）、次数
    pfile.writelines("initial perplexity=%f\n" % pre_perp)
    pfile.writelines('K_index:'+str([np.min(m),np.max(m),mode(m)[0][0],mode(m)[1][0]])+'\n')
    # k最小、最大、众数（int）、次数
    pfile.writelines('A_index:'+str([np.min(acc_value),np.max(acc_value),mode(acc_value)[0][0],
                                     mode(acc_value)[1][0]])+'\n')  # a最小、最大、众数（int）、次数

    pfile.writelines("%d K=%d min_p=%f A=%f\n" % (x[0] + 1, len(result_set[x[0]]) ,p[x[0]], acc_value[x[0]]))
    # 输出最小p值的topic_word结果
    for each in result_set[x[0]]:
        pfile.writelines(toString(each)+'\n')

    if (x[0] + 1) != iteration:
        pfile.writelines("%d K=%d last_p=%f\n" % (iteration, len(result_set[-1]), p[-1]))  # 输出最后一次迭代的topic_word结果
        for each in result_set[-1]:
            pfile.writelines(toString(each)+'\n')

    y = np.argsort(-np.array(acc_value, dtype=float))
    pfile.writelines("%d K=%d p=%f max_A=%f\n" % (y[0] + 1, len(result_set[y[0]]), p[y[0]], acc_value[y[0]]))
    # 输出最小p值的topic_word结果
    for each in result_set[y[0]]:
        pfile.writelines(toString(each)+'\n')
    return pfile


def toString(each):
    string = each.keys()[0] + ':{'
    for ev in each[each.keys()[0]]:
        string += ev.keys()[0].encode('utf-8') + ':' + ev[ev.keys()[0]] + ','
    string += '}'
    return string


def accuracy(k_ji, corpus, label, acc_style):
    d_maxk = []
    if acc_style == 0:                    # 文档分类方式
        d_kw=[]                           # 相同k的权重加和
        for j in xrange(len(k_ji)):
            d_kw.append({})
            ctf = corpus[j]
            if len(ctf) == 0:
                d_maxk.append(None)
                continue
            w_k = np.zeros(len(ctf), dtype=float)
            for i in xrange(len(ctf)):
                w_k[i] = ctf[i][1]
            w_k /= np.sum(w_k)
            # 将第二个数组按第一个数组（有重复值）的映射，权重加和
            k_i = k_ji[j]
            k_unrep = list(set(k_i))      # 取出全部不重复的k
            for k in k_unrep:
                b_i = []
                for i in xrange(len(k_i)):
                    if k == k_i[i]:       # 循环得到重复k的下标
                        b_i.append(i)
                sw = w_k[b_i]
                d_kw[j][k] = np.sum(sw)   # k的权和
            result = sorted(iteritems(d_kw[j]), key=operator.itemgetter(1), reverse=True)[0][0]
            d_maxk.append(result)

    else:   # 以k出现的次数作为文档分类标准
        d_k = [{} for _ in xrange(len(k_ji))]
        for ik, dk in zip(k_ji, d_k):     # 生成d_k矩阵
            for each in ik:
                dk[each] = dk.get(each, 0) + 1
            d = sorted(dk.iteritems(), key=lambda t:t[1], reverse=True)
            d_maxk.append(d[0][0])        # 生产d_maxk列表

    label_class = {}.fromkeys(label).keys()
    r_a = []
    for i in label_class:
        per_c = []
        for j in range(len(label)):
            if label[j] == i:
                per_c.append(j)
        r_a.append(per_c)

    # 在k内部取原正确类文档的个数和
    d_maxk = np.array(d_maxk)             # d_maxk=[2,1,0,4]
    k_upj = {}.fromkeys(d_maxk).keys()    # 取出全部不重复的k
    dict_jk = {}                          # k:[文档]
    k_rp = 0
    for k in k_upj:
        b_i = []
        for i in xrange(len(d_maxk)):
            if k == d_maxk[i]:            # 循环得到重复k的下标
                b_i.append(i)
        if len(b_i) != 1:
            k_rp += 1                     # 计算非单簇的k
        dict_jk[k] = b_i
    acc = 0
    for k in k_upj:                       # 对每个k进行文章类别统计
        cout = {}
        for n in range(len(label_class)):        # 类：文档数
            cout[n] = 0
        for jki in dict_jk[k]:            # 在一个k中进行统计
            for c, rg in enumerate(r_a):
                if jki in rg:
                    cout[c] += 1
        cdict = sorted(cout.iteritems(), key=lambda d: d[1], reverse=True)[0][1]     # 大到小排，获取类文档次数
        acc += cdict
    str_acc = float(acc) / d_maxk.size * (float(k_rp) / len(k_upj))                  # z-measure
    return str_acc
