import sys
import os
import cv2, time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from drawCMC import *
from FaceDetectAndCrop import *
from generateFeatures import *
from drawGIDandROC import *
from compare import *

face_cascade = cv2.CascadeClassifier('haarcascade_default.xml')

def getGenuineImposter(comp, match):
    genuine = []
    imposter = []
    r, c = np.where(match == True)
    # for i in range(r.size):
    #     genuine.append(comp[r[i],c[i]])
    C = np.unique(c)
    for i in range(C.size):
        ind = np.where(c == C[i])
        # print(ind[0])
        genuine.append(comp[r[ind[0][:]], C[i]].min())
    print(len(genuine), min(genuine), max(genuine))
    r, c = np.where(match == False)
    # print(r,c)
    for i in range(r.size):
        imposter.append(comp[r[i], c[i]])
    # C=np.unique(r)
    # for i in range(C.size):
    #     ind = np.where(r == C[i])
    #     # print(ind[0])
    #     imposter.append(np.mean(comp[C[i],c[ind[0][:]]]))
    print(len(imposter), min(imposter), max(imposter))
    # pass
    return genuine, imposter



if __name__ == "__main__":
    paths = ['gallery', 'probe']
    [browseAndSaveFaces(p) for p in paths]

    print('Generating features ...')
    galfeat = generateFeatures('gallery/')
    prfeat = generateFeatures('probe/')
    print('Done!')
    mat = None
    if not os.path.exists('dists.mat'):
        print('Generating Comparison Matrix ...')
        mat = compareFeat(galfeat, prfeat)
        pickle.dump(mat, open('dists.mat', 'wb'))
    else:
        print('Loading Comparison Matrix ...')
        mat = pickle.load(open('dists.mat', 'rb'))
        print('Done!')
    match = None
    if not os.path.exists('match.mat'):
        match = getLabelMat('gallery/', 'probe/')
        pickle.dump(match, open('match.mat', 'wb'))
    else:
        match = pickle.load(open('match.mat', 'rb'))
    # print(match.shape)
    # print(mat.shape)

    genuine, imposter = getGenuineImposter(mat, match)

    drawGI(genuine, imposter)
    plotROC(genuine, imposter)
    plotCMC(mat, match)
    # print(mat.max(),mat.min())
