import numpy as np # linear algebra
import pandas as pd

import os

from sklearn import datasets,svm,metrics

mnist_train = pd.read_csv("C:\Users\coco1\Desktop\AI시스템설계\train.csv") # train csv 파일 불러오기
mnist_test = pd.read_csv("C:\Users\coco1\Desktop\AI시스템설계\test.csv") # test csv 파일 불러오기

train = mnist_train.values
test = mnist_test.values

train_data=train[:,1:] # csv 파일의 읽어올 범위 지정
train_labels=train[:,0]

test_data=test[:,1:]
test_labels=test[:,0]

train_data=train_data/255
test_data=test_data/255

#c = 부동값, kernel은 알고리즘에서 사용할 커널 유형 결정, gamma는 커널에 대한 계수, probability = False는 빈 배열이라는 뜻
classifier=svm.SVC(C=200,kernel='rbf',gamma=0.01,probability=False)
classifier.fit(train_data,train_labels)

predicted=classifier.predict(test_data)

print("Classification report for classifier :\n%s\n" % (metrics.classification_report(test_labels, predicted)))