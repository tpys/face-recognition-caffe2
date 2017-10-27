from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate


class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths


def get_dataset(paths, has_class_directories=True):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            image_paths = get_image_paths(facedir)
            dataset.append(ImageClass(class_name, image_paths))
    return dataset


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return np.array(image_paths_flat), np.array(labels_flat, dtype=np.int)


def filter_dataset(dataset, min_num_images=10):
    filtered_dataset = dataset
    removelist = []
    for i in range(len(filtered_dataset)):
        if len(filtered_dataset[i].image_paths) < min_num_images:
            removelist.append(i)
    ix = sorted(list(set(removelist)), reverse=True)
    for i in ix:
        del(filtered_dataset[i])

    return filtered_dataset


def train_test_split(data_dir, train_test_ratio=0.8, min_num_images_per_class=10):
    data_set = get_dataset(data_dir)
    data_set = filter_dataset(data_set, min_num_images_per_class)
    print('num_class:{}'.format(len(data_set)))

    image_list, label_list = get_image_paths_and_labels(data_set)

    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    # train_index, test_index = sss.split(image_list, label_list)

    num_images = len(image_list)
    num_training = int(num_images * train_test_ratio)
    random_index = np.random.permutation(num_images)
    labels = label_list[random_index]
    images = image_list[random_index]

    train_images = images[:num_training]
    train_labels = labels[:num_training]
    test_images = images[num_training:]
    test_labels = labels[num_training:]

    return images, labels, train_images, train_labels, test_images, test_labels


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    dist = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
    indices = np.arange(nrof_pairs)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])
        print ("best threshold: {:f}".format(thresholds[best_threshold_index]))

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy



def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    dist = np.sum(np.multiply(embeddings1, embeddings2), axis=1)

    indices = np.arange(nrof_pairs)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_roc_matlab_version(embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])

    thresh_num = 10000
    k_fold = KFold(n_splits=nrof_folds, shuffle=True)
    accuracy = np.zeros((nrof_folds))

    dist = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
    indices = np.arange(nrof_pairs)
    print (nrof_pairs)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        best_threshold = get_threshold(dist[train_set], actual_issame[train_set], thresh_num)
        print (best_threshold)
        _, _, accuracy[fold_idx] = calculate_accuracy(best_threshold, dist[test_set], actual_issame[test_set])
    return accuracy


def get_threshold(scores, flags, thresh_num):
    accuracy = np.zeros((2 * thresh_num + 1))
    thresholds = np.array([x / float(thresh_num) for x in range(-thresh_num, thresh_num + 1)])
    for idx in range(len(thresholds)):
        _, _, accuracy[idx] = calculate_accuracy(thresholds[idx], scores, flags)
    best_threshold = np.mean(thresholds[accuracy == np.max(accuracy)])
    return best_threshold


