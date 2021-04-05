import MeCab
import csv
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import itertools

# MeCabオブジェクトの生成
mt = MeCab.Tagger('')
mt.parse('')

# トピック数の設定
NUM_TOPICS = 3

if __name__ == "__main__":
    # トレーニングデータの読み込み
    # train_texts は二次元のリスト
    # テキストデータを一件ずつ分かち書き（名詞、動詞、形容詞に限定）して train_texts に格納するだけ
    train_texts = []
    with open('./split_sympton_tweets/troublesome.txt', 'r') as f:
        for line in f:
            text = []
            node = mt.parseToNode(line.strip())
            while node:
                fields = node.feature.split(",")
#                if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
                if fields[0] == '名詞'   :
#                if fields[0] == '形容詞':
#                if fields[0] == '動詞'  :
                    text.append(node.surface)
                node = node.next
            train_texts.append(text)

    print(train_texts)
    X = list(itertools.chain.from_iterable(train_texts))
    print(X)
    # 単語の出現頻度データを作成
    tf_vectorizer = CountVectorizer(max_df=0.8, min_df=10, stop_words='english')
    tf = tf_vectorizer.fit_transform(X)
    len(tf_vectorizer.get_feature_names())

    # LDAのモデル作成と学習
    lda = LatentDirichletAllocation(
                            n_components=5,
                            learning_method='online',
                            max_iter=20
                        )
    print("set")
    lda.fit(tf)
    print("set2")
    features = tf_vectorizer.get_feature_names()

    count = 0
    jj = 0
    comp = []
    text = ""
    for i, component in enumerate(lda.components_[:5]):
        print("component:", i)
        idx = component.argsort()[::-1][:10]
        for j in idx:
            count += component[j]
            comp.append(component[j])

        for j in idx:
            text += features[j] + "," + str(comp[jj]/count) + "\n"
            print(features[j], comp[jj]/count)
            jj += 1
        jj = 0
        count = 0
#        for k in range(10):
#            comp.pop(k)

    with open("しんどい.csv","w") as f:
        f.write(text)
