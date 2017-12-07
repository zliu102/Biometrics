import os
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_default.xml')

def browseAndSaveFaces(base):
    # images = []
    imfile = open(base+'\images.txt','w')
    lblfile = open(base+'\labels.txt','w')
    labels = []
    for i, (dirname, dirnames, filenames) in enumerate(os.walk(base)):
        if dirname == base:
            labels = [d for d in dirnames]
        count = 0
        for file in filenames:
            if base == 'probe':
                if file.endswith('.jpg') and not file.startswith('face'):
                    img = cv2.imread(os.path.join(dirname,file))
                    if saveFace(img,dirname,file):
                        imfile.write(os.path.join(dirname,'face_'+file)+'\n')
                        lblfile.write(labels[i-1]+'\n')
                        print('face_'+file)
                        break
            else:
                if file.endswith('.jpg') and not file.startswith('face'):
                    img = cv2.imread(os.path.join(dirname,file))
                    if saveFace(img,dirname,file):
                        imfile.write(os.path.join(dirname,'face_'+file)+'\n')
                        lblfile.write(labels[i-1]+'\n')
                        print('face_'+file)
                        count += 1
                        # if count == 4:
                        #     break
    imfile.close()
    lblfile.close()


def saveFace(img, path='_', imname='_'):
    img_color = img  # cv2.imread('image.jpg')
    h, w = img_color.shape[:2]
    ratio = w / float(h)
    img_color = cv2.resize(img_color, (int(ratio * 720), 720))  # frame from camera
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    faces = detect(img_gray)
    if len(faces) == 0:
        return False
    for rect in faces:
        xy1 = get_xy(faces)
        if xy1:
            fac = 30
            const = 6
            const2 = int(const / 2)
            cropi = img_gray[xy1[2] + fac + const:xy1[3] - fac, xy1[0] + fac + const2:xy1[1] - fac - const2]
            cropp = cv2.resize(cropi, (100, 100))
            cropp = process_image(cropp)
            fname = 'face_'
            cv2.imwrite(path + '/' + fname + imname, cropp)
    return True

def detect(img, cascade=face_cascade):
    img = cv2.equalizeHist(img)
    rects = cascade.detectMultiScale(img, minSize=(200, 200), flags=1, scaleFactor=1.2)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def get_xy(rects):
    for x1, y1, x2, y2 in rects:
        xy = [x1, x2, y1, y2]
        return xy

def process_image(imag):
    gr = correct_gamma(imag, 0.3)
    gr = cv2.equalizeHist(gr)
    return gr


def correct_gamma(img, correction):
    img = img / 255.0
    img = cv2.pow(img, correction)
    return np.uint8(img * 255)
