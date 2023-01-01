import numpy as np
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

Potato_healthy_dir, Potato_early_blight_dir, Potato_Late_blight_dir = "D:\Downloads (chrome)\plant disease datasets\PlantVillage\Potato___healthy", "D:\Downloads (chrome)\plant disease datasets\PlantVillage\Potato___Early_blight", "D:\Downloads (chrome)\plant disease datasets\PlantVillage\Potato___Late_blight"
Potato_healthy_dir2, Potato_early_blight_dir2, Potato_Late_blight_dir2 = "D:\Downloads (chrome)\plant disease datasets\plantTwo\PLD_3_Classes_256\Training\Healthy", "D:\Downloads (chrome)\plant disease datasets\plantTwo\PLD_3_Classes_256\Training\Early_Blight", "D:\Downloads (chrome)\plant disease datasets\plantTwo\PLD_3_Classes_256\Training\Late_Blight"
#Potato_healthy_dir, Potato_early_blight_dir, Potato_Late_blight_dir = "PlantVillage/Potato___healthy", "PlantVillage/Potato___Early_blight", "PlantVillage/Potato___Late_blight"
#Potato_healthy_dir2, Potato_early_blight_dir2, Potato_Late_blight_dir2 = "PLD_3_Classes_256/Training/Healthy", "PLD_3_Classes_256/Training/Early_Blight", "PLD_3_Classes_256/Training/Late_Blight"
n = 200


def saveModelToFile(model, filename):
    pickle.dump(model, open(filename, 'wb'))


class result:
    def __init__(self, acc, f1, time):
        self.accuracies = acc
        self.f1 = f1
        self.time = time


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

    # counter = 0
    # for image in os.listdir(Potato_healthy_dir2):
    #     current = cv.imread(Potato_healthy_dir2 + "/" + image)
    #     Potato_healthy.append(current)
    #     Potato_labels_1.append(0)
    #     counter += 1
    #     if counter == n:
    #         break
    #
    # counter = 0
    # for image in os.listdir(Potato_early_blight_dir2):
    #     current = cv.imread(Potato_early_blight_dir2 + "/" + image)
    #     Potato_EB.append(current)
    #     Potato_labels_2.append(1)
    #     counter += 1
    #     if counter == n:
    #         break
    #
    # counter = 0
    # for image in os.listdir(Potato_Late_blight_dir2):
    #     current = cv.imread(Potato_Late_blight_dir2 + "/" + image)
    #     Potato_LB.append(current)
    #     Potato_labels_3.append(2)
    #     counter += 1
    #     if counter == n:
    #         break

    Potato_healthy_np = np.array(Potato_healthy)
    Potato_EB_np = np.array(Potato_EB)
    Potato_LB_np = np.array(Potato_LB)
    # print(Potato_LB_np.shape)

    # Potato_healthy_extracted_features = []
    # for i in range(Potato_healthy_np.shape[0]):
    #     image = Potato_healthy_np[i]
    #     image_g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #     orb = cv.ORB_create(nfeatures=1500)
    #     kp, des = orb.detectAndCompute(image_g, None)
    #     curr_features = []
    #     for count, j in enumerate(kp):
    #         curr_features.append(j.pt)
    #         if count == 500:
    #             break
    #     Potato_healthy_extracted_features.append(curr_features)
    #
    # Potato_EB_extracted_features = []
    # for i in range(Potato_EB_np.shape[0]):
    #     image = Potato_EB_np[i]
    #     image_g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #     orb = cv.ORB_create(nfeatures=1500)
    #     kp, des = orb.detectAndCompute(image_g, None)
    #     curr_features = []
    #     for count, j in enumerate(kp):
    #         curr_features.append(j.pt)
    #         if count == 500:
    #             break
    #     Potato_EB_extracted_features.append(curr_features)
    #
    # Potato_LB_extracted_features = []
    # for i in range(Potato_LB_np.shape[0]):
    #     image = Potato_LB_np[i]
    #     image_g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #     orb = cv.ORB_create(nfeatures=1500)
    #     kp, des = orb.detectAndCompute(image_g, None)
    #     curr_features = []
    #     for count, j in enumerate(kp):
    #         curr_features.append(j.pt)
    #         if count == 500:
    #             break
    #     Potato_LB_extracted_features.append(curr_features)
    #
    # PF1 = np.array(Potato_healthy_extracted_features)
    # PF2 = np.array(Potato_EB_extracted_features)
    # PF3 = np.array(Potato_LB_extracted_features)
    #
    # X, y = np.concatenate((PF1, PF2, PF3)), np.array(Potato_labels_1 + Potato_labels_2 + Potato_labels_3)
    #
    # X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    X = np.concatenate((Potato_healthy_np, Potato_EB, Potato_LB))
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
    y = np.array(Potato_labels_1 + Potato_labels_2 + Potato_labels_3)
    # X = X[:250, :]
    # y = y[:250]
    print(X.shape)

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    # print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)
    accuracies = []
    f1_scores = []
    times = []

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    begin = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    print(accuracy_score(y_pred, y_test))
    accuracies.append(accuracy_score(y_pred, y_test))
    f1_scores.append(f1_score(y_pred, y_test, average="weighted"))
    times.append(end - begin)

    # param_grid = {'C': [0.1, 1, 10, 100, 1000],
    #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #               'kernel': ['rbf']}
    model = SVC()
    model.fit(X_train, y_train)
    # grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    # grid.fit(X_train, y_train)

    begin = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    # print best parameter after tuning
    # print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    # print(grid.best_estimator_)

    """ Results : {'C': 0.1, 'gamma': 1, 'kernel': 'rbf'}    SVC(C=0.1, gamma=1)"""

    print(accuracy_score(y_pred, y_test))
    accuracies.append(accuracy_score(y_pred, y_test))
    f1_scores.append(f1_score(y_pred, y_test, average="weighted"))
    times.append(end - begin)

    model = LogisticRegression(multi_class='multinomial', max_iter=2000)
    model.fit(X_train, y_train)
    # grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
    # model = LogisticRegression(max_iter=10000)
    # model_cv = GridSearchCV(model, grid, cv=10)
    # model_cv.fit(X_train, y_train)

    begin = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    # print("tuned hpyerparameters :(best parameters) ", model_cv.best_params_)
    # print("accuracy :",  model_cv.best_score_)

    print(accuracy_score(y_pred, y_test))
    accuracies.append(accuracy_score(y_pred, y_test))
    f1_scores.append(f1_score(y_pred, y_test, average="weighted"))
    times.append(end - begin)

    # saveModelToFile(model, "model.ml")

    model = BernoulliNB()
    model.fit(X_train, y_train)

    begin = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    print(accuracy_score(y_pred, y_test))
    accuracies.append(accuracy_score(y_pred, y_test))
    f1_scores.append(f1_score(y_pred, y_test, average="weighted"))
    times.append(end - begin)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    begin = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    print(accuracy_score(y_pred, y_test))
    accuracies.append(accuracy_score(y_pred, y_test))
    f1_scores.append(f1_score(y_pred, y_test, average="weighted"))
    times.append(end - begin)

    res = {
        "accuracies": accuracies,
        "f1_scores": f1_scores,
        "times": times
    }
    # saveModelToFile(res, "results.rsl")
    # print(res)

    x = ["DecisionTree", "SVM", "Logistic Regression", "BernoulliNB", "Random Forest"]

    x_axis = np.arange(len(x))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x_axis - width / 2, accuracies, width, label='Accuracy')
    rects2 = ax.bar(x_axis + width / 2, f1_scores, width, label='f1_score')
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    ax.set_ylabel('Metric Values')
    ax.set_title('Machine Learning Model Evaluation Metrics')
    ax.set_xticks(x_axis, x)
    ax.legend()

    fig.tight_layout()

    plt.show()

    plt.bar(x, times)
    plt.ylabel("prediction time")
    plt.show()

    # plt.bar(x, accuracies)
    # plt.show()
    #
    # plt.bar(x, f1_scores)
    # plt.show()

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
