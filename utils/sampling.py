#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid_train_test(dataset_train,dataset_test, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs_train,num_imgs_test = 2000, 30, 5
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs_train = np.arange(num_shards*num_imgs_train)
    idxs_test = np.arange(num_shards * num_imgs_test)
    labels_train = dataset_train.train_labels.numpy()
    labels_test = dataset_test.train_labels.numpy()

    # sort labels
    idxs_labels_train = np.vstack((idxs_train, labels_train))
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_train = idxs_labels_train[:, idxs_labels_train[1, :].argsort()]
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_train = idxs_labels_train[0, :]
    idxs_test = idxs_labels_test[0, :]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 20, replace=False))
        #idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs_train[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[rand * num_imgs_test:(rand + 1) * num_imgs_test]), axis=0)
    return dict_users, dict_users_test
def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        rand_set = set([20*(i%(num_users-1)),20*(i%(num_users-1))+30])
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users





if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
