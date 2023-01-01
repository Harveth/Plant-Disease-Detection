import cv2 as cv
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5 import uic
from stylsheets import *

#img_path = "D:\Downloads (chrome)\plant disease datasets\PlantVillage\Potato___Late_blight\/feefc118-4434-4ffb-afbb-02fb292f72b6___RS_LB 2874.JPG"
img_path = "PlantVillage/Potato___Late_blight/feefc118-4434-4ffb-afbb-02fb292f72b6___RS_LB 2874.JPG"

classes = {0: "Healthy", 1: "Early Blight", 2: "Late Blight"}


def convert_image_to_feature_vector(image):
    return image.reshape(image.shape[0] * image.shape[1] * image.shape[2]).reshape(1, -1)


def load_model():
    return pickle.load(open('model.ml', 'rb'))

def load_results():
    return pickle.load(open('results.rsl', 'rb'))


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # self.setWindowFlag(Qt.FramelessWindowHint)
        uic.loadUi('ui_files/gui.ui', self)
        self.image_path = ""
        self.model = load_model()
        self.setFixedSize(1000, 450)
        self.setWindowTitle("Plant Disease Detection")
        self.pushButton.clicked.connect(self.SelectImage)
        self.pushButton_2.clicked.connect(self.classifyImage)
        self.pushButton_3.clicked.connect(self.showGraphs)

    def loadImage(self, image):
        self.imgLBL.setPixmap(QPixmap.fromImage(image))

    def SelectImage(self):
        fname = QFileDialog.getOpenFileName(self, "Select Image", "", "All Files (*)")
        if fname:
            self.image_path = fname[0]
            print(self.image_path)
            image = cv.imread(self.image_path)
            QtFormat = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            scaled = QtFormat.scaled(481, 591, Qt.KeepAspectRatio)
            self.loadImage(scaled)

    def showGraphs(self):
        res = load_results()
        accuracies,precision,recall, f1_scores = res["accuracies"], res["precision"],res["recall"],res["f1_scores"]
        print(res)
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

        # plt.bar(x, times)
        # plt.ylabel("time")


    def classifyImage(self, image):
        if self.image_path != "":
            plant_image = cv.imread(self.image_path)
            fv = convert_image_to_feature_vector(plant_image)
            print(fv.shape)
            pred = self.model.predict(fv)
            print(classes[pred[0]])
            self.label_4.setText("Classified as : " + classes[pred[0]])


if __name__ == "__main__":
    # model = load_model()
    # plant_image = cv.imread(img_path)
    # fv = convert_image_to_feature_vector(plant_image)
    # print(fv.shape)
    # pred = model.predict(fv)
    # print(classes[pred[0]])
    # plant_image = cv.putText(plant_image, classes[pred[0]], (10, 250), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
    # cv.imshow("image", plant_image)
    # cv.waitKey(0)
    App = QApplication(sys.argv)
    App.setStyleSheet(stylesheet1)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())
