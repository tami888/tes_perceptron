# coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt

def train(wvec, xvec, label) :
    low = 0.5       # 学習係数
    if (np.dot(wvec,xvec) * label < 0):     # 誤分類した場合更新する
        wvec_new = wvec + label * low * xvec
        return wvec_new
    else:
        return wvec

if __name__ == '__main__':

    train_num = 100     # 学習データ数


    #class1の学習データ
    x1_1=np.random.rand(train_num/2) * 5 + 1            # x成分 (1から6の乱数を50個生成)
    x1_2=np.random.rand(int(train_num/2)) * 5 + 1       # y成分 (1から6の乱数を50個生成)
    label_x1 = np.ones(train_num/2)                     # ラベル（すべて1）

    #class2の学習データ
    x2_1=(np.random.rand(train_num/2) * 5 + 1) * -1     # x成分 (1から6の乱数を50個生成し、マイナスにする)
    x2_2=(np.random.rand(train_num/2) * 5 + 1) * -1     # y成分 (1から6の乱数を50個生成し、マイナスにする)
    label_x2 = np.ones(train_num/2) * -1                # ラベル（すべて-1）

    x0=np.ones(train_num/2) # x0は常に1
    # print x0
    x1=np.c_[x0, x1_1, x1_2]
    # print x1
    x2=np.c_[x0, x2_1, x2_2]
    # print x2

    xvecs=np.r_[x1, x2]
    labels = np.r_[label_x1, label_x2]

    wvec = np.array([2,-1,2])                       # 初期の重みベクトル 適当に決める


    for j in range(100):
        for xvec, label in zip(xvecs, labels):      # zipで要素を同時にループ
            wvec = train(wvec, xvec, label)
            print wvec
    print "test"
    print wvec


    plt.scatter(x1[:,1], x1[:,2], c='blue', marker="o")     # x1のxとyでプロット
    plt.scatter(x2[:,1], x2[:,2], c='red', marker="o")      # x2のxとyでプロット

    #境界線
    x_fig = np.array(range(-8, 8))
    #print x_fig
    y_fig = -(wvec[1]/wvec[2])*x_fig - (wvec[0]/wvec[2])
    #print y_fig
    plt.plot(x_fig,y_fig)
    plt.show()
