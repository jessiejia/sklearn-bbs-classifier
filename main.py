# -*- coding: utf-8 -*-
import sys
import jieba
import numpy
import json
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

from flask import Flask, request, make_response

app = Flask(__name__)

class Classify:
    comma_tokenizer = lambda x: jieba.cut(x, cut_all=True)
    clf = None
    train_data0 = 'train_data0.txt'
    train_data1 = 'train_data1.txt'

    def __init__(self, modelpath='tmpVector.pkl'):
        self.modelpath = modelpath

    def vectorize(self, data):
        vectorizer = HashingVectorizer(tokenizer=lambda x: jieba.cut(x, cut_all=True), n_features=30000,
                                       non_negative=True)
        v_data = vectorizer.fit_transform(data)
        return v_data

    def getClf(self):
        if Classify.clf is None:
            try:
                print '--reading joblib-'
                Classify.clf = joblib.load(self.modelpath)
            except Exception:
                self.updateClf()

        return Classify.clf

    def updateClf(self):
        # 从文件加载
        print '--reading files--'
        train_data0 = train_data1 = []
        try:
            with open(Classify.train_data0, 'r') as myfile:
                train_data0 = myfile.readlines()
        except Exception:
            pass
        try:
            with open(Classify.train_data1, 'r') as myfile:
                train_data1 = myfile.readlines()
        except Exception:
            pass
        train_data = train_data0 + train_data1
        train_tags = [0] * len(train_data0) + [1] * len(train_data1)

        clf = MultinomialNB(alpha=0.01)
        clf.fit(self.vectorize(train_data), numpy.asarray(train_tags))
        joblib.dump(clf, self.modelpath)
        Classify.clf = clf

    def train(self, new_train_data, train_tag):
        # 追加到存储文件中
        file = "train_data" + str(train_tag) + ".txt"
        with open(file, 'a') as myfile:
            myfile.write(new_train_data.encode('utf8') + '\n')
        # 重新生成clf
        self.updateClf()

    def predict(self, test_data):
        if self.getClf() is not None:
            pred = Classify.clf.predict(self.vectorize(test_data))
            return pred
        return []


def jsonSuccessReturn(data=''):
    return make_response(json.dumps({'status': 'success', 'result': data}))
def jsonFailReturn(data=''):
    return make_response(json.dumps({'status': 'fail', 'result': {'msg':data}}))
classify = Classify();

@app.route('/')
def index():
    return 'hello world! info'


@app.route('/predict/')
def predict():
    test_data = request.args.get('content')
    tags = classify.predict([test_data])
    if len(tags) > 0:
        return jsonSuccessReturn({'is_spam':tags[0]})
    return jsonFailReturn('')


@app.route('/train/')
def train():
    train_data = request.args.get('content')
    type = request.args.get('type', 'spam')
    train_tags = 1 if type == 'spam' else 0
    classify.train(train_data, train_tags)
    return jsonSuccessReturn()


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception, e:
        port = 5000

    app.run(host='0.0.0.0', port=port, debug=True)
