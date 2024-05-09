# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm, trange

class NaiveBayesianClassifier:
    def load(self, data, label):
        self.data = data
        self.label = label
        self.features = self.feature_normalization(self.data)
        self.data_point = self.features.shape[0]
        self.feature_num = self.features.shape[1]
        self.label_num = max(self.label) + 1
        self._get_parameters()

    def add_data(self, data, label):
        data = np.concatenate([self.data, data])
        label = np.concatenate([self.label, label])
        self.load(data, label)

    def feature_normalization(self, data):
        mean = np.mean(self.data, axis = 0)
        std = np.std(self.data, axis = 0)
        
        return (data - mean) / std

    def _get_parameters(self):
        self.prior = np.zeros([self.label_num])
        self.mu = np.zeros([self.label_num, self.feature_num])
        self.sigma = np.zeros([self.label_num, self.feature_num])
        classes = [np.empty([self.feature_num, 0]) for _ in range(self.label_num)]

        print("calculate prior, each class's mean and standard variation")
        for idx, item in tqdm(enumerate(self.features)):
            self.prior[self.label[idx]] += 1
            classes[self.label[idx]] = np.append(classes[self.label[idx]], item.reshape(self.feature_num, 1), axis = 1)

        for i in trange(self.label_num):
            self.prior /= self.prior[i].sum()
            self.mu[i] = classes[i].mean(axis = 1)
            self.sigma[i] = classes[i].std(axis = 1)
        

    def Gaussian_Log_PDF(self, x, mu, sigma): 
        return (np.square(x - mu) / (np.square(sigma)) + np.log(2 * np.pi * np.square(sigma))) / -2
    
    def classify(self, value):
        data_point = value.__len__()    
        posterior = np.zeros([data_point, self.label_num])
        
        features = self.feature_normalization(value)

        for i in range(self.label_num):
            likelihood = self.Gaussian_Log_PDF(features, self.mu[i], self.sigma[i])
            posterior[:, i] = self.prior[i] * likelihood.sum(axis = 1) 
        return np.argmax(posterior, axis = 1)
            
    def accuracy(self, pred, gnd):
        data_point = len(gnd)
        hit_num = np.sum(pred == gnd)

        return (hit_num / data_point) * 100, hit_num