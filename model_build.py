import numpy as np
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

Potato_healthy_dir, Potato_early_blight_dir, Potato_Late_blight_dir = "D:\Downloads (chrome)\plant disease datasets\PlantVillage\Potato___healthy", "D:\Downloads (chrome)\plant disease datasets\PlantVillage\Potato___Early_blight", "D:\Downloads (chrome)\plant disease datasets\PlantVillage\Potato___Late_blight"
n = 200


def saveModelToFile(model, filename):
    pickle.dump(model, open(filename, 'wb'))


if __name__ == "__main__":
    Potato_labels_1, Potato_labels_2, Potato_labels_3, Potato_healthy, Potato_EB, Potato_LB = [], [], [], [], [], []
    counter = 0
    for image in os.listdir(Potato_healthy_dir):
        current = cv.imread(Potato_healthy_dir + "/" + image)
        Potato_healthy.append(current)
        Potato_labels_1.append(0)
        counter += 1
        if counter == n:
            break

    counter = 0
    for image in os.listdir(Potato_early_blight_dir):
        current = cv.imread(Potato_early_blight_dir + "/" + image)
        Potato_EB.append(current)
        Potato_labels_2.append(1)
        counter += 1
        if counter == n:
            break

    counter = 0
    for image in os.listdir(Potato_Late_blight_dir):
        current = cv.imread(Potato_Late_blight_dir + "/" + image)
        Potato_LB.append(current)
        Potato_labels_3.append(2)
        counter += 1
        if counter == n:
            break

    Potato_healthy_np = np.array(Potato_healthy)
    Potato_EB_np = np.array(Potato_EB)
    Potato_LB_np = np.array(Potato_LB)
    # print(Potato_LB_np.shape)

    Potato_healthy_extracted_features = []
    for i in range(Potato_healthy_np.shape[0]):
        image = Potato_healthy_np[i]
        image_g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create(nfeatures=1000)
        kp, des = orb.detectAndCompute(image_g, None)
        curr_features = []
        for count, j in enumerate(kp):
            curr_features.append(j.pt)
            if count == 500:
                break
        Potato_healthy_extracted_features.append(curr_features)

    Potato_EB_extracted_features = []
    for i in range(Potato_EB_np.shape[0]):
        image = Potato_EB_np[i]
        image_g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create(nfeatures=1000)
        kp, des = orb.detectAndCompute(image_g, None)
        curr_features = []
        for count, j in enumerate(kp):
            curr_features.append(j.pt)
            if count == 500:
                break
        Potato_EB_extracted_features.append(curr_features)

    Potato_LB_extracted_features = []
    for i in range(Potato_LB_np.shape[0]):
        image = Potato_LB_np[i]
        image_g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create(nfeatures=1500)
        kp, des = orb.detectAndCompute(image_g, None)
        curr_features = []
        for count, j in enumerate(kp):
            curr_features.append(j.pt)
            if count == 500:
                break
        Potato_LB_extracted_features.append(curr_features)

    print(np.array(Potato_healthy_extracted_features).shape)
    print(np.array(Potato_EB_extracted_features).shape)
    print(np.array(Potato_LB_extracted_features).shape)

    PF1 = np.array(Potato_healthy_extracted_features)
    PF2 = np.array(Potato_EB_extracted_features)
    PF3 = np.array(Potato_LB_extracted_features)

    X, y = np.concatenate((PF1, PF2, PF3)), np.array(Potato_labels_1 + Potato_labels_2 + Potato_labels_3)

    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)
    accuracies = []

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(accuracy_score(y_pred, y_test))
    accuracies.append(accuracy_score(y_pred, y_test))



    model = SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(accuracy_score(y_pred, y_test))
    accuracies.append(accuracy_score(y_pred, y_test))



    model = LogisticRegression(multi_class='multinomial', max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(accuracy_score(y_pred, y_test))
    accuracies.append(accuracy_score(y_pred, y_test))



    model = BernoulliNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(accuracy_score(y_pred, y_test))
    accuracies.append(accuracy_score(y_pred, y_test))


    x = ["DecisionTree", "SVM", "Logistic Regression", "BernoulliNB"]

    plt.bar(x, accuracies)
    plt.show()




    # print(len(Potato_labels_1) * len(Potato_labels_2) * len(Potato_labels_3) * 256 * 256 * 3)
    # print(Potato_healthy_np[0].reshape(Potato_healthy_np.shape[1] * Potato_healthy_np.shape[2] * Potato_healthy_np.shape[3]).shape)

    # test_image = Potato_healthy_np[8]
    # test_image_gs = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
    #
    # orb = cv.ORB_create(nfeatures=2000)
    # kp, des = orb.detectAndCompute(test_image_gs, None)
    #
    # kp_image = cv.drawKeypoints(test_image, kp, None, color=(0, 255, 0), flags=0)
    #
    # cv.imshow("test", test_image_gs)
    # cv.imshow("orb", kp_image)
    # cv.waitKey(0)
