import cv2 as cv
import pickle

img_path = "D:\Downloads (chrome)\plant disease datasets\PlantVillage\Potato___Late_blight\/feefc118-4434-4ffb-afbb-02fb292f72b6___RS_LB 2874.JPG"
classes = {0: "Healthy", 1: "Early Blight", 2: "Late Blight"}


def convert_image_to_feature_vector(image):
    return image.reshape(image.shape[0] * image.shape[1] * image.shape[2]).reshape(1, -1)


def load_model():
    return pickle.load(open('model.ml', 'rb'))


if __name__ == "__main__":
    model = load_model()
    plant_image = cv.imread(img_path)
    fv = convert_image_to_feature_vector(plant_image)
    print(fv.shape)
    pred = model.predict(fv)
    print(classes[pred[0]])
    plant_image = cv.putText(plant_image, classes[pred[0]], (10, 250), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
    cv.imshow("image", plant_image)
    cv.waitKey(0)

