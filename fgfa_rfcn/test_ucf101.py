# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import _init_paths
import numpy as np
import cv2
import argparse
import pandas as pd
import os
import sys
import pprint
import time
import logging
import shutil
import dill
import pickle as pkl
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='test R-FCN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import mxnet as mx
from symbols import *
from function.test_rcnn import test_rcnn
from utils.create_logger import create_logger
from core.loader_ucf101 import TestLoader
from core.tester import Predictor, pred_eval, pred_eval_multiprocess
from utils.load_model import load_param
from utils.load_data_ucf101 import load_gt_imdb

def get_predictor(sym, sym_instance, cfg, arg_params, aux_params, test_data, ctx):
    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (16, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('heatmap', (16, 1, max([v[0] for v in [(15, 20)]]), max([v[1] for v in [(15, 20)]]))),
                       ]]

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor


def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.iloc[i, data.columns.get_loc(1)])
    return labels

def test_net(cfg, dataset, image_set, ctx, prefix, epoch, shuffle):
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    prefix = os.path.join(final_output_path, prefix)


    # print config
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load dataset and prepare imdb for training
    config.dataset.dataset = 'UCF101'
    config.dataset.root_path = '/data/waybarrios/FGFA_CACHE'
    config.dataset.dataset_path = '/data/weik/UCF101/JPG/'
    config.dataset.traintestlist_path = '/data/weik/UCF101/ucfTrainTestList/'
    config.TRAIN.FLIP = False
    split = '01'
    classes = load_labels(os.path.join(config.dataset.traintestlist_path, 'classInd.txt'))
    gtdb, gtviddb = load_gt_imdb(config.dataset.dataset, config.dataset.root_path, config.dataset.dataset_path,
                                 config.dataset.traintestlist_path, split=split, subset='train', flip=config.TRAIN.FLIP)


    # load symbol
    config.symbol = 'resnet_v1_101_flownet_rfcn_ucf101'
    shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), final_output_path)
    feat_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()

    # load model
    epoch = 20
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    feat_sym = feat_sym_instance.get_test_symbol(cfg)

    cache_path = os.path.join(config.dataset.root_path, 'cache_predict', 'split' + split)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    final_outputs = []
    count_tp = 0
    for sample_idx, roidb in enumerate(gtviddb):
        if sample_idx<3000:
            continue
        if sample_idx>=6000:
            break
        print('{0}--->{1}'.format(sample_idx, roidb['video_name']))
        # get test data iter
        results = {
            'video': roidb['video_name'],
            'clips': []
        }

        test_datas = TestLoader([roidb], cfg, batch_size=1, shuffle=shuffle, ctx=[ctx[0]])
        for nbatch, data_batch in enumerate(test_datas):
            feat_predictors = get_predictor(feat_sym, feat_sym_instance, cfg, arg_params, aux_params, test_datas, [ctx[0]])
            output = feat_predictors.predict(data_batch)[0]
            results['clips'].append(output['softmax_output'].asnumpy())
        
        prediction = np.asanyarray(results['clips'])
        import ipdb; ipdb.set_trace()
        prediction = np.transpose(prediction, [0,2,1])
        prediction_index = np.argmax(np.average(prediction, axis=0))

        prediction_name = classes[prediction_index]
        print('GT:{0}--->Prediction:{1}'.format(roidb['video_name'], prediction_name))
        final_outputs.append(results)

        
     #with open(os.path.join(cache_path, 'test.pkl'), 'wb') as f:
      #       pkl.dump(final_outputs, f)


def main():
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    print args

    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)

    test_net(config, config.dataset.dataset, config.dataset.test_image_set, ctx, config.TRAIN.model_prefix, config.TEST.test_epoch, shuffle=False)


if __name__ == '__main__':
    main()
