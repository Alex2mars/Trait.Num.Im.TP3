import os.path
import random as rnd
from os import listdir
from os.path import join

import cv2
import keras
import keras.layers
import keras.layers
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications import resnet_v2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load and preprocess images -> Retruns : labels ["label1", "label2", ...] and data [img1, img2, img3, ...] with imgX a nparray
# Preprocess : Grayscale to -> 3 channels | + preprocess to ResNet101V2 input
def load_dataset(categorical=True):
    path = "dataset"
    labels = []
    dataset = []
    # For each folder corresponding to labels
    label_number = 0
    for folder_class in listdir(path):
        folder_path = join(path, folder_class)
        for img_name in listdir(folder_path):
            # Corresponding label added to labels
            if categorical:
                labels.append(folder_class)
            else:
                labels.append(label_number)

            # Load image as NP array
            img_path = join(folder_path, img_name)
            img_np = np.asarray(Image.open(img_path))

            # Greyscale to RGB
            img_np = np.expand_dims(img_np, -1)
            img_np = np.repeat(img_np, 3, axis=2)

            # Transform images to ResNet101v2 inputs
            img_input = resnet_v2.preprocess_input(img_np)
            dataset.append(img_input)

        label_number += 1

    return labels, dataset

#not used
def balance_data(data_per_class):
    n_data = max(map(lambda kv: len(kv[1]), data_per_class.items()))
    # Tirage avec remise pour dupliquer les données
    for label, class_data in data_per_class.items():
        n_dataclass = len(class_data)
        while len(class_data) < n_data:
            to_duplicate_index = rnd.randint(0, n_dataclass)
            class_data.append(class_data[to_duplicate_index].copy())


def transfer_learning_model():
    base_model = resnet_v2.ResNet101V2(
        include_top=False,
        input_shape=(64, 64, 3),
        weights="imagenet",
    )

    avg = keras.layers.GlobalAvgPool2D()(base_model.output)
    output = keras.layers.Dense(4, activation="softmax")(avg)
    model = keras.Model(inputs=base_model.input, outputs=output)

    return base_model, model


def transfer_learning_compile_model(model):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


def transfer_learning_train(base_model, global_model, X_train, y_train, X_test, y_test):
    # Freeze base layers
    base_model.trainable = False
    transfer_learning_compile_model(global_model)

    global_model.fit(x=np.asarray(X_train), y=np.asarray(y_train), batch_size=20, epochs=10,
                     validation_data=(np.asarray(X_test), np.asarray(y_test)))



def custom_cnn_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(keras.layers.Flatten())
    # hidden layer
    model.add(keras.layers.Dense(100, activation='relu'))
    # output layer
    model.add(keras.layers.Dense(4, activation='softmax'))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

def custom_cnn_train(model, X_train, y_train, X_test, y_test):
    model.fit(x=np.asarray(X_train), y=np.asarray(y_train), batch_size=50, epochs=30, validation_data=(np.asarray(X_test), np.asarray(y_test)))

#not used
def dataset_to_dict(labels, data):
    data_per_class = dict()

    for name in np.unique(labels):
        data_per_class[name] = []

    for id_img, img in enumerate(data):
        label_name = labels[id_img]
        data_per_class[label_name].append(img)

    return data_per_class

#not used
def dict_to_dataset(data_dict):
    labels = []
    data = []
    for key, val in data_dict.items():
        for img in val:
            labels.append(key)
            data.append(img)
    return labels, data


def input_to_img(input_data):
    return ((input_data + 1) * 127.999).astype(np.uint8)


def input_data_pred(input_data, label_encoder, model):
    one_hot_result = model.predict(np.array([input_data, ]))[0]
    return label_encoder.inverse_transform([np.argmax(one_hot_result)])[0]


def main():
    print("Loading dataset...")
    labels, data = load_dataset()
    '''
    print("Balancing dataset...")
    data_per_class = dataset_to_dict(labels, data)
    balance_data(data_per_class)
    labels, data = dict_to_dataset(data_per_class)
    '''

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)

    print("Séparation en jeux de données train, test, validation...")
    X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25)

    ### Transfer learning model ###
    if not os.path.isdir("tl_model"):
        tl_base_model, tl_model = transfer_learning_model()
        transfer_learning_train(tl_base_model, tl_model, X_train, y_train, X_test, y_test)
        tl_model.save("tl_model")
    else:
        print("Loading TL model from folder...")
        tl_model = keras.models.load_model("tl_model")

    print("[TL] Calcul accuracy pour dataset de validation...")
    y_pred = tl_model.predict(np.asarray(X_valid))
    y_pred = np.argmax(y_pred, axis=1)
    print("[TL] Validation accuracy score : ", accuracy_score(y_valid, y_pred))

    ### Custom CNN model ###
    if not os.path.isdir("custom_model"):
        custom_model = custom_cnn_model()
        custom_cnn_train(custom_model, X_train, y_train, X_test, y_test)
        custom_model.save("custom_model")
    else:
        print("Loading custom model from folder...")
        custom_model = keras.models.load_model("custom_model")

    print("[CUSTOM CNN] Calcul accuracy pour dataset de validation...")
    y_pred = custom_model.predict(np.asarray(X_valid))
    y_pred = np.argmax(y_pred, axis=1)
    print("[CUSTOM CNN] Validation accuracy score : ", accuracy_score(y_valid, y_pred))

    validation_model = custom_model

    for img_input, label in zip(X_valid, y_valid):
        cv2.imshow("Photo en cours d'analyse", input_to_img(img_input))
        print(input_data_pred(img_input, label_encoder, validation_model))
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
