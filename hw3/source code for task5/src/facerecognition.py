import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os, sys
import random, math
from scipy import linalg
from collections import defaultdict

import config as cfg

def showImgs(imgs, titles=None):
    '''Show images, 3 per row'''
    nrows = math.ceil(len(imgs) / 3)
    plt.figure(figsize=(12, nrows * 3))
    gs = gridspec.GridSpec(nrows, 3, width_ratios=[1]*3)
    for i in range(len(imgs)):
        plt.subplot(gs[i]), plt.imshow(imgs[i], 'gray', aspect='equal')
        plt.xticks([]), plt.yticks([])
        if titles is not None:
            plt.title(titles[i])
    plt.show()

def clipfaces(imgs, face_cascade):
    faces = [None] * len(imgs)
    for i in range(len(imgs)):
        gray = np.array(imgs[i], dtype=np.uint8)
        face = face_cascade.detectMultiScale(gray, 1.1, 5)
        x,y,w,h = face[0]
        clip_ratio = cfg.CLIP_RATIO
        x, y = int(x + w * (1 - clip_ratio) / 2), int(y + h * (1 - clip_ratio))
        w, h = int(w * clip_ratio), int(h * clip_ratio)
        faces[i] = gray[y:y+h,x:x+w]
    return [cv2.resize(face, (50, 50)) for face in faces]

gamma = cfg.PP_GAMMA
gammaCorrection = lambda I, gamma: (I / 255.0) ** gamma * 255

sigma0, sigma1 = cfg.PP_SIGMA1, cfg.PP_SIGMA2
DOG = lambda I,sgm0,sgm1: cv2.GaussianBlur(I,(0,0),sgm0) - cv2.GaussianBlur(I,(0,0),sgm1)

alpha, tau = cfg.PP_ALPHA, cfg.PP_TAU
CE0 = lambda I, a, t: I / ((np.array([x for x in np.nditer(abs(I)) if x < t]) ** a).mean() ** (1/a))
CE1 = lambda I, a: I / ((abs(I) ** a).mean() ** (1/a))

def preprocess(img, gamma=0.4, sigma0=1, sigma1=2, alpha=0.1, tau=10):
    gc = gammaCorrection(img, gamma)
    dog = DOG(gc, sigma0, sigma1)
    ce = CE1(CE0(dog, alpha, tau), alpha)
    return ce

def predict(fvecs_tr, labels_tr, fvecs_te, sim_func=np.dot, k = 6):
    # k nearest-neighbor
    prediction = []
    for vec in fvecs_te:
        sim = sorted([(sim_func(vec, fvecs_tr[j]), labels_tr[j]) 
                      for j in range(len(labels_tr))], key=lambda x: x[0], reverse=True)[:k]
        scores = defaultdict(float)
        for score, label in sim:
            scores[label] += score
        prediction.append(max(scores.items(), key=lambda x: x[1])[0])
    return prediction

def precision(prediction, labels):
    tp = 0
    for i in range(len(prediction)):
        tp += prediction[i] == labels[i]
    return tp / len(labels)

def main():
    if len(sys.argv) < 2:
        print('Usage: %s <path of image files> \n (e.g. data/yalefaces)' % sys.argv[0])
        return
    img_dir = sys.argv[1] + '/'
    print(img_dir)
    execpath = sys.argv[0]
    fdpath = execpath[:execpath.rfind('facerecognition.py')] + 'haarcascade_frontalface_default.xml'

    # read and clip images

    img_files = [filename for filename in os.listdir(img_dir) if filename.endswith('.jpg')]
    imgs = [cv2.imread(img_dir + filename, 0) for filename in img_files]

    face_cascade = cv2.CascadeClassifier(fdpath)
    faces = clipfaces(imgs, face_cascade)

    rd3imgs = random.sample(faces, k=3)
    print('3 random images:')
    showImgs(rd3imgs)

    # preprocess

    faces = [preprocess(img, gamma, sigma0, sigma1, alpha, tau) for img in faces]
    print('After preprocessing:')
    rd3imgs = [preprocess(img, gamma, sigma0, sigma1, alpha, tau) for img in rd3imgs]
    showImgs(rd3imgs)

    # Split dataset into training and testing datasets

    n_classes = cfg.N_CLASSES
    n_ipc = cfg.N_IMAGES_PER_CLASS
    n_tspc = cfg.N_TRAINING_SAMPLES_PER_CLASS

    train_idxes = [] # 6:5
    for i in range(n_classes):
        train_idxes += random.sample(range(i*n_ipc,i*n_ipc+n_ipc), k=n_tspc)
    test_idxes = [i for i in range(len(faces)) if i not in train_idxes]

    faces_train = [faces[i] for i in train_idxes]
    faces_test = [faces[i] for i in test_idxes]

    # Generate image vectors

    h, w = faces[0].shape
    nDims = h * w
    vecs_train = np.array([img.reshape(nDims) for img in faces_train])
    vecs_test = np.array([img.reshape(nDims) for img in faces_test])

    # Perform PCA

    face_mean, eigenvecs = cv2.PCACompute(vecs_train, None)

    print('mean face:')
    plt.imshow(face_mean.reshape((h, w)), 'gray')
    plt.show()
    print('first 9 eigenfaces:')
    showImgs(eigenvecs.reshape((eigenvecs.shape[0], h, w))[:9,:,:])

    # Eigenfaces recognition

    ndims_ef = cfg.EF_N_FEATURE_DIMS_PRESERVED
    nvecs_train = vecs_train - np.stack(list(face_mean)*len(vecs_train))
    nvecs_test = vecs_test - np.stack(list(face_mean)*len(vecs_test))
    fvecs_train = nvecs_train.dot(eigenvecs.T)
    fvecs_test = nvecs_test.dot(eigenvecs.T) # row vectors

    # normalized feature vectors
    nfvecs_train = np.stack([vec / ((vec ** 2).sum() ** 0.5) for vec in fvecs_train[:,:ndims_ef]])
    nfvecs_test = np.stack([vec / ((vec ** 2).sum() ** 0.5) for vec in fvecs_test[:,:ndims_ef]])

    ef_prd = predict(nfvecs_train, [idx // n_ipc for idx in train_idxes], nfvecs_test)
    print('Recognition precison using eigenfaces:')
    print(precision(ef_prd, [idx // n_ipc for idx in test_idxes]))

    # Construct k-nearest-neighbor graph

    N = len(fvecs_train)
    nn_amat = np.zeros((N, N), dtype=np.float64)
    k = cfg.LF_K_NEAREST_NEIGHBORS # 4, 7
    for i in range(N):
        nns = sorted([(((fvecs_train[i] - fvecs_train[j])**2).sum(), j) 
                      for j in range(N)], key=lambda x: x[0])[:k+1]
        for dist, j in nns:
            nn_amat[i,j] = nn_amat[j,i] = dist
        nn_amat[i,i] = 0

    # Assign weight to each edge in the graph

    t = nn_amat.sum() / (nn_amat > 0).sum()

    S = np.zeros(nn_amat.shape)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if nn_amat[i,j] != 0:
                S[i,j] = math.e**(-nn_amat[i,j]/t)

    # Laplacianfaces recognition

    ldims_pca = cfg.LF_N_DIMS_PCA
    D = np.diag(S.sum(axis=0))
    L = D - S
    X = fvecs_train.T[:ldims_pca,:]
    evals, Wlpp = linalg.eigh(X.dot(L).dot(X.T), X.dot(D).dot(X.T))

    ndims_lf = cfg.LF_N_FEATURE_DIMS_PRESERVED
    Wlpp = Wlpp[:,:ndims_lf]
    lfvecs_train = fvecs_train[:,:ldims_pca].dot(Wlpp)
    lfvecs_test = fvecs_test[:,:ldims_pca].dot(Wlpp)

    nlfvecs_train = np.stack([vec / ((vec ** 2).sum() ** 0.5) for vec in lfvecs_train])
    nlfvecs_test = np.stack([vec / ((vec ** 2).sum() ** 0.5) for vec in lfvecs_test])

    lf_prd = predict(nlfvecs_train, [idx // n_ipc for idx in train_idxes], nlfvecs_test)
    print('Recognition precison using Laplacianfaces:')
    print(precision(lf_prd, [idx // n_ipc for idx in test_idxes]))

    # Fisherfaces recognition

    Sb, Sw = np.zeros((nDims, nDims)), np.zeros((nDims, nDims))
    for i in range(n_classes):
        mui = vecs_train[i*n_tspc:(i+1)*n_tspc,:].mean(axis=0)
        for j in range(n_tspc):
            Sw += np.outer(vecs_train[i*n_tspc+j,:]-mui, vecs_train[i*n_tspc+j,:]-mui)
        Sb += np.outer(mui - face_mean, mui - face_mean) * n_tspc

    fdims_pca = cfg.FF_N_DIMS_PCA
    Wpca = eigenvecs[:fdims_pca,:].T
    evals, Wfld = linalg.eigh(Wpca.T.dot(Sb).dot(Wpca), Wpca.T.dot(Sw).dot(Wpca))

    ffvecs_train = fvecs_train[:,:fdims_pca].dot(Wfld[:,1-n_classes:])
    ffvecs_test = fvecs_test[:,:fdims_pca].dot(Wfld[:,1-n_classes:])
    nffvecs_train = np.stack([vec / ((vec ** 2).sum() ** 0.5) for vec in ffvecs_train])
    nffvecs_test = np.stack([vec / ((vec ** 2).sum() ** 0.5) for vec in ffvecs_test])

    ff_prd = predict(nffvecs_train, [idx // n_ipc for idx in train_idxes], nffvecs_test)
    print('Recognition precison using Fisherfaces:')
    print(precision(ff_prd, [idx // n_ipc for idx in test_idxes]))

if __name__ == '__main__':
    main()