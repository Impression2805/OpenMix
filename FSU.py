import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from scipy.spatial import distance
from scipy.stats import chi2
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import normalize
import argparse
import os
import csv
import math
import pandas as pd
import numpy as np
import resource
from collections import OrderedDict

from model import resnet
from model import resnet18
from model import densenet_BC
from model import vgg
from model import mobilenet
from model import efficientnet
from model import wrn
from model import convmixer
from utils import data as dataset
from utils import crl_utils
from utils import metrics
from utils import utils
import train_cvpr
from utils.data_utils import *

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def csv_writter(path, dic, start):
    if os.path.isdir(path) == False: os.makedirs(path)
    os.chdir(path)
    if start == 1:
        mode = 'w'
    else:
        mode = 'a'
    with open('logs.csv', mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        if start == 1:
            writer.writerow(dic.keys())
        writer.writerow([elem["string"] for elem in dic.values()])

class Counter(dict):
    def __missing__(self, key):
        return None


def dist_feature(mode, features, target_labels): ##function to compute FSU
    features_locs = []
    for lab in np.unique(target_labels):
        features_locs.append(np.where(target_labels == lab)[0])

    if mode == 'intra':
        if isinstance(features, torch.Tensor):
            intrafeatures = features.detach().cpu().numpy()
        else:
            intrafeatures = features

        intra_dists = []
        for loc in features_locs:
            c_dists = distance.cdist(intrafeatures[loc], intrafeatures[loc], 'cosine')
            c_dists = np.sum(c_dists)/(len(c_dists)**2-len(c_dists))
            intra_dists.append(c_dists)
        intra_dists = np.array(intra_dists)
        maxval      = np.max(intra_dists[1-np.isnan(intra_dists)])
        intra_dists[np.isnan(intra_dists)] = maxval
        intra_dists[np.isinf(intra_dists)] = maxval
        dist_metric = dist_metric_intra = np.mean(intra_dists)

    if mode == 'inter':
        if not isinstance(features, torch.Tensor):
            coms = []
            for loc in features_locs:
                com   = normalize(np.mean(features[loc], axis=0).reshape(1,-1)).reshape(-1)
                coms.append(com)
            mean_inter_dist = distance.cdist(np.array(coms), np.array(coms), 'cosine')
            dist_metric = dist_metric_inter = np.sum(mean_inter_dist)/(len(mean_inter_dist)**2-len(mean_inter_dist))
        else:
            coms = []
            for loc in features_locs:
                com   = torch.nn.functional.normalize(torch.mean(features[loc], dim=0).reshape(1, -1), dim=-1).reshape(1,-1)
                coms.append(com)
            mean_inter_dist = 1-torch.cat(coms, dim=0).mm(torch.cat(coms, dim=0).T).detach().cpu().numpy()
            dist_metric = dist_metric_inter = np.sum(mean_inter_dist)/(len(mean_inter_dist)**2-len(mean_inter_dist))

    if mode == 'fsu':
        dist_metric = dist_metric_intra/np.clip(dist_metric_inter, 1e-8, None)
    return dist_metric


parser = argparse.ArgumentParser(description='OpenMix: Exploring Outlier Samples for Misclassification Detection')
parser.add_argument('--epochs', default=200, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--plot', default=10, type=int, help='')
parser.add_argument('--run', default=3, type=int, help='')
parser.add_argument('--classnumber', default=10, type=int, help='class number for the dataset')
parser.add_argument('--classnumber_aug', default=1, type=int, help='class number for the dataset')
parser.add_argument('--data', default='cifar10', type=str, help='Dataset name to use [cifar10, cifar100]')
parser.add_argument('--model', default='res110', type=str, help='Models name to use [res110, wrn, dense, resnet18, vgg, cmixer]')
parser.add_argument('--method', default='OpenMix_manifold', type=str, help='[OpenMix, OpenMix_manifold, OpenMix-CRL, '
                                                                           'classAug, Mixup, Manifold, RegMixup, '
                                                                           'OE, Baseline, CRL')
parser.add_argument('--data_path', default='/data/datasets/', type=str, help='Dataset directory')
parser.add_argument('--save_path', default='./output/', type=str, help='Savefiles directory')
parser.add_argument('--rank_weight', default=1.0, type=float, help='Rank loss weight')
parser.add_argument('--gpu', default='6', type=str, help='GPU id to use')
parser.add_argument('--lambda_o', type=float, default=1, help='[0.1, 0.5, 1.0, 1.5, 2] dnl loss weight')
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-aux_set', type=str, default='RandomImages', help='RandomImages')
parser.add_argument('-aux_size', type=int, default=-1, help='using all RandomImages data')
parser.add_argument('-prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('-aux_batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--openness', type=float, default=0.5, help='[0.1, 0.5, 1.0, 1.5, 2]')


args = parser.parse_args()

def main():
    acc_list=[]
    auroc_list=[]
    aupr_success_list=[]
    aupr_list=[]
    fpr_list=[]
    aurc_list=[]
    eaurc_list=[]
    ece_list=[]
    nll_list=[]
    brier_list=[]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True
    save_path = args.save_path + args.data + '_' + args.model + '_' + args.method
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_loader, valid_loader, test_loader, \
    test_onehot, test_label = dataset.get_loader(args.data, args.data_path, args.batch_size, args)

    if args.data == 'cifar100':
        num_class = 100
        args.classnumber = 100
    else:
        args.classnumber = 10
        num_class = 10

    if args.method == 'OpenMix' or args.method == 'OpenMix-CRL' or args.method == 'OpenMix_manifold':
        num_class = num_class + 1

    if args.method == 'classAug':
        if args.data == 'cifar100':
            args.newclassnum = 900
            num_class = num_class + args.newclassnum
        else:
            args.newclassnum = 45
            num_class = num_class + args.newclassnum

    model_dict = {
        "num_classes": num_class,
    }

    save_path = args.save_path + args.data + '_' + args.model + '_' + args.method
    save_path_model = save_path + '/'
    model_state_dict = torch.load(os.path.join(save_path_model, 'model.pth'))
    model.load_state_dict(model_state_dict)

    model.eval()
    feature_list = []
    labels_list = []

    for input, target, idx in test_loader:
        input = input.cuda()
        target = target.long().cuda()
        _, feature = model(input, feature_output=True)
        feature_list.append(feature.detach().cpu())
        labels_list.append(target)

    features = torch.cat(feature_list)
    labels = torch.cat(labels_list)

    dist_intra = dist_feature('intra', features, labels)
    dist_inter = dist_feature('inter', features, labels)
    FSU = dist_feature('fsu', features, labels)
    print(dist_intra, dist_inter, FSU)


if __name__ == "__main__":
    main()


