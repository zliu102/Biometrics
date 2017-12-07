import sys
import os
import cv2, time
import numpy as np
import pickle
import matplotlib.pyplot as plt

def generateFeatures(path = 'gallery/'):
    surf = cv2.xfeatures2d.SURF_create(500)
    imgs = open(path+'images.txt','r').readlines()
    # labels = open(path+'labels.txt','r').readlines()
    imgs = [im.strip('\n') for im in imgs]
    # labels = [lbl.strip('\n') for lbl in labels]
    features = []
    print(path)
    x = float(len(imgs))
    for i, im in enumerate(imgs):
        img = cv2.imread(im,0)
        features.append(surf.detectAndCompute(img,None))
        print('\rCompleted: {:.2f}%'.format(((i+1)/x)*100),end = ' ')
    print()
    return features