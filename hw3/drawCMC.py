import sys
import os
import cv2, time
import numpy as np
import pickle
import matplotlib.pyplot as plt

def plotCMC(mat, matches, gpath='gallery/'):
    labels = open(gpath + 'labels.txt', 'r').readlines()
    labels = [lbl.strip() for lbl in labels]
    labarr = np.asarray(labels)
    # print(labarr.shape)
    R = np.unique(labarr).tolist()
    cli = len(R)
    fig = plt.figure()
    ma = None
    match = None
    r, c = np.where(matches == True)
    C = np.unique(c).tolist()
    que = len(C)
    ma = np.ndarray((cli, que), dtype=np.float32)
    match = np.ndarray((cli, que), dtype=np.bool_)

    for i, row in enumerate(R):
        ind_r = np.where(labarr == row)
        for j, col in enumerate(C):
            # ind_c = np.where(c == col)

            temp = mat[ind_r[0], col]
            temp_m = matches[ind_r[0], col]

            idx = np.argsort(temp, axis=0)
            ma[i, j] = temp[idx[0]]
            match[i, j] = temp_m[idx[0]]

    ind = np.argsort(ma, axis=0)
    # print(ind)
    r, c = ind.shape
    x = list(range(1, r + 1))
    y = []
    for i in range(r):
        count = 0.0
        for j in C:
            comp = ma[ind[:i, j], j]
            m = match[ind[:i, j], j]
            z = np.where(m == True)
            if z[0].size > 0:
                count += 1
        y.append(count / que)
    plt.plot(x, y)
    plt.xlabel('Rank Counted as Recognition')
    plt.ylabel('Recognition Rate')
    plt.title('CMC Curve')
    plt.xlim(1, r)
    fig.savefig('CMCCurve.png')
    plt.show()