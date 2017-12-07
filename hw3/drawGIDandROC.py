import sys
import os
import cv2, time
import numpy as np
import pickle
import matplotlib.pyplot as plt

def drawGI(genuine, imposter):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(0.0, 1.0, num=101)
    # print(x)
    yg = []
    yi = []
    for i in range(x.size):
        r = np.where(genuine <= x[i])
        if i > 0:
            yg.append(r[0].size - sum(yg))
        else:
            yg.append(r[0].size)
        r = np.where(imposter <= x[i])
        if i > 0:
            yi.append(r[0].size - sum(yi))
        else:
            yi.append(r[0].size)
    # print(yg,yi)
    y1 = [float(i) / sum(yg) for i in yg]
    # bins = len(set(genuine))
    # print(bins)
    plt.plot(x, y1, color='green', label='Genuine')
    y2 = [float(i) / sum(yi) for i in yi]
    plt.plot(x, y2, color='red', label='Imposter')
    plt.axvline(x=0.3849, linestyle='dashed', color='black', label='Threshold = 0.385')
    plt.xlabel('Face Match Scores')
    plt.ylabel('Relative Frequency')
    plt.title('Genuine and Imposter distributions')
    plt.xlim(0.0, 1.0)
    plt.legend()
    fig.savefig('GenuineImposter.png')
    plt.show()


def plotROC(genuine, imposter):
    fig = plt.figure()
    rng = np.linspace(0, 1.0, num=101)
    tot_g = len(genuine)
    tot_i = len(imposter)
    TPR = []
    FPR = []
    for i in range(101):
        r = np.where(genuine <= rng[i])
        TPR.append(r[0].size / float(tot_g))
        r = np.where(imposter <= rng[i])
        FPR.append(r[0].size / float(tot_i))

    plt.plot(FPR, TPR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    # plt.axes([0.0,1.0,0.0,1.0])
    # fig.savefig('ROCPlot.png')
    plt.show()