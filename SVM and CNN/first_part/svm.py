import cv2
import numpy as np
import os

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def extractingData():
    """
    """
    with open('dataset/annotation.txt', 'r') as f:
        data = f.readlines()
        names = [x.split(' ')[0] for x in data]
        targets = [x.split(' ')[1] for x in data]

    images = []
    imagesSmall = []
    labels = []
    for root, dirs, files in os.walk("dataset/images"):
        for file in files:
            imagepath = os.path.join(root, file)
            if file.endswith('.jpg'):
                img = cv2.imread(imagepath)
                img2 = cv2.resize(img, (56, 56))
                imagesSmall.append(img2)
                images.append(img)
                labels.append(targets[names.index(file)])

    return imagesSmall, labels, images

if __name__ == '__main__':
    images, labels, reals = extractingData()

    cell = 2
    size = 56
    nbin = 4
    featureVector = (size/cell) * (size/cell) * nbin
    hog = cv2.HOGDescriptor(_winSize=(cell, cell),
                                _blockSize=(cell, cell),
                                _blockStride=(cell, cell),
                                _cellSize=(cell, cell),
                                _nbins=nbin, _histogramNormType = 0, _gammaCorrection = True)

    features = []
    for img in images:
        features.append(hog.compute(img).reshape(featureVector))


    X = np.asarray(features)
    y = np.asarray(labels)

    print X.shape
    print y.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print accuracy_score(y_test, y_pred)

    for x in range(391, 420):
        hogImg = np.asarray(hog.compute(images[x])).reshape(-1, featureVector)
        pred = clf.predict(hogImg)
        print pred
        cv2.imshow("S", reals[x])
