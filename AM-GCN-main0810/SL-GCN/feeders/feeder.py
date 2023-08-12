import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import random
sys.path.extend(['../'])
from feeders import tools

flip_index = np.concatenate(([0,2,1,4,3,6,5],[17,18,19,20,21,22,23,24,25,26],[7,8,9,10,11,12,13,14,15,16]), axis=0) 

class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, random_mirror=False, random_mirror_p=0.5, is_vector=False):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.random_mirror = random_mirror
        self.random_mirror_p = random_mirror_p
        self.load_data()
        self.is_vector = is_vector
        if normalization:
            self.get_mean_map()
        print(len(self.label))

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
            
        if self.random_mirror:
            if random.random() > self.random_mirror_p:
                assert data_numpy.shape[2] == 27
                data_numpy = data_numpy[:,:,flip_index,:]
                if self.is_vector:
                    data_numpy[0,:,:,:] = - data_numpy[0,:,:,:]
                else: 
                    data_numpy[0,:,:,:] = 512 - data_numpy[0,:,:,:]

        if self.normalization:
            # data_numpy = (data_numpy - self.mean_map) / self.std_map
            assert data_numpy.shape[0] == 3
            if self.is_vector:
                data_numpy[0,:,0,:] = data_numpy[0,:,0,:] - data_numpy[0,:,0,0].mean(axis=0)
                data_numpy[1,:,0,:] = data_numpy[1,:,0,:] - data_numpy[1,:,0,0].mean(axis=0)
            else:
                data_numpy[0,:,:,:] = data_numpy[0,:,:,:] - data_numpy[0,:,0,0].mean(axis=0)
                data_numpy[1,:,:,:] = data_numpy[1,:,:,:] - data_numpy[1,:,0,0].mean(axis=0)

        if self.random_shift:
            if self.is_vector:
                data_numpy[0,:,0,:] += random.random() * 20 - 10.0
                data_numpy[1,:,0,:] += random.random() * 20 - 10.0
            else:
                data_numpy[0,:,:,:] += random.random() * 20 - 10.0
                data_numpy[1,:,:,:] += random.random() * 20 - 10.0

        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

