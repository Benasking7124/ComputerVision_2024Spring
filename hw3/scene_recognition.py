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


def compute_dsift(img, stride, size):
    # To do
    sift = cv2.SIFT_create()
    keypoints = [cv2.KeyPoint(x, y, size) for y in range(0, img.shape[0], stride) for x in range(0, img.shape[1], stride)]
    _, dense_feature = sift.compute(img, keypoints)
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
    for i in range(nbrs.shape[0]):
        for j in range(k):
            class_counter[label_train[nbrs[i][j]]] += 1
        label_test_pred.append(max(class_counter, key=lambda key: class_counter[key]))
        for j in class_counter:
            class_counter[j] = 0
    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    # Create training img list
    train_list = []
    for x in img_train_list:
        img = cv2.imread(x, 0)
        tiny = get_tiny_image(img, 16)
        train_list.append(tiny)
    train_list = np.array(train_list)
    nsamples, nx, ny = train_list.shape
    train_list.resize((nsamples, nx * ny))

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
    label_prediction_list = predict_knn(train_list, label_train_list, test_list, 5)

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
    data_list = np.concatenate([np.split(array, array.shape[0]) for array in dense_feature_list])
    data_list = data_list.reshape(data_list.shape[0], data_list.shape[2])
    print("start")
    import time
    before = time.time()
    print(before)
    kmeans = KMeans(n_clusters = dict_size).fit(data_list)
    duration = time.time() - before
    print(duration)
    vocab = kmeans.cluster_centers_
    np.savetxt("vocabulary.txt", vocab)
    return vocab


def compute_bow(feature, vocab):
    # To do
    bow_feature = np.zeros((vocab.shape[0]))
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(vocab)
    _, index = knn.kneighbors(feature)
    index = index.reshape(-1)

    for i in index:
        bow_feature[i] += 1
    bow_feature /= np.linalg.norm(bow_feature)

    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    # Create training feature list
    train_list = []
    for x in img_train_list:
        img = cv2.imread(x, 0)
        dsift = compute_dsift(img, 5, 10)
        train_list.append(dsift)

    # Create testing feature list
    test_list = []
    for x in img_test_list:
        img = cv2.imread(x, 0)
        dsift = compute_dsift(img, 5, 10)
        test_list.append(dsift)

    # Build Vocabulary
    vocabularies = build_visual_dictionary(train_list, 50)
    # vocabularies = np.loadtxt("vocabulary.txt")

    # Compute BoW list
    train_bow_list = []
    for x in train_list:
        train_bow_list.append(compute_bow(x, vocabularies))

    test_bow_list = []
    for x in test_list:
        test_bow_list.append(compute_bow(x, vocabularies))
    
    # Train KNN
    n_neighbor = 10
    knn = NearestNeighbors(n_neighbors = n_neighbor, algorithm = 'ball_tree').fit(train_bow_list)
    _, nbrs = knn.kneighbors(test_bow_list)

    # create class counter
    class_counter = {}
    for x in label_train_list:
        class_counter[x] = 0

    # Count label number to predict
    label_prediction_list = []
    for i in range(nbrs.shape[0]):
        for j in range(n_neighbor):
            class_counter[label_train_list[nbrs[i][j]]] += 1
        label_prediction_list.append(max(class_counter, key=lambda key: class_counter[key]))
        for j in class_counter:
            class_counter[j] = 0

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

    # img = cv2.imread("scene_classification_data/test/Bedroom/image_0003.jpg", 0)
    # dsift = compute_dsift(img, 10, 3)
    # compute_bow(dsift, voc)
    
    # classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)




