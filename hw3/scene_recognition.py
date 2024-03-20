import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_dsift(img):
    # To do
    return dense_feature


def get_tiny_image(img, output_size):
    # To do
    img = cv2.resize(img, (output_size, output_size))
    feature = (img - np.mean(img)) / np.std(img)
    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    # To do
    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(feature_train)
    _, nbrs = knn.kneighbors(feature_test)

    # create class counter
    class_counter = {}
    for x in label_train:
        class_counter[x] = 0

    label_test_pred = []
    for i in range(feature_test.shape[0]):
        for j in range(k):
            class_counter[label_train[nbrs[i][j]]] += 1
        label_test_pred.append(max(class_counter, key=lambda key: class_counter[key]))
        for j in class_counter:
            class_counter[j] = 0
    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    # Create training img list
    feature_list = []
    for x in img_train_list:
        img = cv2.imread(x, 0)
        tiny = get_tiny_image(img, 16)
        feature_list.append(tiny)
    feature_list = np.array(feature_list)
    nsamples, nx, ny = feature_list.shape
    feature_list.resize((nsamples, nx * ny))

    # Create testing img list
    test_list = []
    for x in img_test_list:
        img = cv2.imread(x, 0)
        tiny = get_tiny_image(img, 16)
        test_list.append(tiny)
    test_list = np.array(test_list)
    nsamples, nx, ny = test_list.shape
    test_list.resize((nsamples, nx * ny))

    # Predict KNN
    label_prediction_list = predict_knn(feature_list, label_train_list, test_list, 5)

    # Counting number of data for each label
    numbers_of_data = [0] * len(label_classes)
    confusion = np.zeros((len(label_classes), len(label_classes)))
    for i in range(len(label_test_list)):
        index_correct_class = label_classes.index(label_test_list[i])
        index_predict_class = label_classes.index(label_prediction_list[i])

        numbers_of_data[index_correct_class] += 1
        confusion[index_correct_class][index_predict_class] += 1

    # Calculate Confusion Matrix and accuracy
    accuracy = 0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            confusion[i][j] /= numbers_of_data[i]
            if (i == j):
                accuracy += confusion[i][j]
    accuracy /= len(label_classes)
    
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dict_size):
    # To do
    return vocab


def compute_bow(feature, vocab):
    # To do
    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    # To do
    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")

    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)




