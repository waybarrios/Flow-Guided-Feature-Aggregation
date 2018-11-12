# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xizhou Zhu, Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import numpy as np
import mxnet as mx
from mxnet.executor_manager import _split_input_slice
import cv2
import math
import os
import random

from utils.image import tensor_vstack, resize, resize_keep_ratio

class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size=1, shuffle=False, ctx=None, work_load_list=None,):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.cfg = config
        self.ori_roidb = roidb
        assert(len(self.ori_roidb) == 1)  # only support one samples in testing by @weik
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list

        # infer properties from roidb
        self.size = self.get_size()
        self.index = np.arange(self.size)

        # get clip-based roidb
        self.roidb = self.get_clip_roidb()

        # decide data and label names (only for training)
        self.data_name = ['data',]#'heatmap']
        self.label_name = ['label1','label2']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = []

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [None for _ in range(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return None

    def get_size(self):
        return int(max(0, math.floor((self.ori_roidb[0]['length'] - self.cfg.sample_duration) / self.cfg.sample_step)) + 1)

    def get_clip_roidb(self):
        clip_roidbs = []
        for idx in self.index:
            clip_roidb = {}
            clip_roidb['images']  = self.ori_roidb[0]['images'][idx*self.cfg.sample_step:idx*self.cfg.sample_step+self.cfg.sample_duration]
            clip_roidb['label']   = self.ori_roidb[0]['label']
            clip_roidb['length']  = self.cfg.sample_duration
            clip_roidb['start']   = idx*self.cfg.sample_step+1
            clip_roidb['end']     = int(idx*self.cfg.sample_step+self.cfg.sample_duration)
            clip_roidb['height']  = self.ori_roidb[0]['height']
            clip_roidb['width']   = self.ori_roidb[0]['width']
            clip_roidb['flipped'] = self.ori_roidb[0]['flipped']

            clip_roidbs.append(clip_roidb)

        return clip_roidbs

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size

            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)
        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(self.parfetch(iroidb))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array([label[key]]) for key in self.label_name] for label in all_label]


    def parfetch(self, iroidb):
        # get testing data for multigpu

        imgs, hms, labels, roidb = self.get_video_label(iroidb, self.cfg)

        #im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

        data = {'data': imgs} # , 'heatmap': hms}
        label = {'label1': labels,'label2':labels}

        return {'data': data, 'label': label}

    def transform(self, im, pixel_means):
        """
        transform into mxnet tensor
        substract pixel size and transform to correct format
        :param im: [height, width, channel] in BGR
        :param pixel_means: [B, G, R pixel means]
        :return: [batch, channel, height, width]
        """
        im_tensor = np.zeros((3, im.shape[0], im.shape[1]))
        for i in range(3):
            im_tensor[i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
        return im_tensor

    def transform_norm(self, im):
        """
        transform into mxnet tensor
        substract pixel size and transform to correct format
        :param im: [height, width, channel] in BGR
        :param pixel_means: [B, G, R pixel means]
        :return: [batch, channel, height, width]
        """
        im_tensor = np.zeros((1, im.shape[0], im.shape[1]))
        im_tensor[0, :, :] = im[:, :] / 255.0

        return im_tensor

    def check_roidb(self, riodb, duration):
        out = range(len(riodb['images']))
        tmpriodb = riodb.copy()

        for index in out:
            if len(tmpriodb['images']) >= duration:
                break

            tmpriodb['images'].append(tmpriodb['images'][index])
            out.append(index)

        return tmpriodb

    def get_video_label(self, iroidb, config):
        """
        preprocess all frames and their hearmap for one video and return processed roidb
        :param roidb: a list of roidb
        :return: list of img as in mxnet format
        roidb add new item['im_info']
        0 --- x (width, second dim of im)
        |
        y (height, first dim of im)
        """
        n_batch = len(iroidb)
        assert(n_batch == 1)
        iroidb = iroidb[0]

        iroidb = self.check_roidb(iroidb, self.cfg.sample_duration)

        num_images = len(iroidb['images'])
        processed_ims = []
        processed_hms = []
        processed_roidb = iroidb.copy()
        processed_labels = int(iroidb['label'])-1

        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im_info = []

        for i in range(num_images):
            assert os.path.exists(iroidb['images'][i]), '%s does not exist'.format(iroidb['images'][i])
            im = cv2.imread(iroidb['images'][i], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if iroidb['flipped']:
                im = im[:, ::-1, :]
            im, im_scale = resize_keep_ratio(im, target_size, max_size)
            im_tensor = self.transform(im, config.network.PIXEL_MEANS)
            processed_ims.append(im_tensor)
            im_info.append([im_tensor.shape[1], im_tensor.shape[2], im_scale])

            # heat-map loading
            #target_size = 15
            #max_size = 20
            heatmap_path = iroidb['images'][i].replace('JPG', 'HeatMap')
            assert os.path.exists(heatmap_path), '%s does not exist'.format(heatmap_path)
            im_heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
            if iroidb['flipped']:
                im = im[:, ::-1, :]
            im_heatmap, im_heatmap_scale = resize_keep_ratio(im_heatmap, 15, 20)
            im_heatmap_tensor = self.transform_norm(im_heatmap)
            processed_hms.append(im_heatmap_tensor)

        processed_roidb['im_info'] = im_info

        return processed_ims, processed_hms, processed_labels, processed_roidb


class TrainLoader(mx.io.DataIter):

    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :param normalize_target: normalize rpn target
        :param bbox_mean: anchor target mean
        :param bbox_std: anchor target std
        :return: AnchorLoader
        """
        super(TrainLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        self.data_name = ['data' ] #, 'heatmap']
        self.label_name = ['label1','label2']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_individual()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_individual()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0


    def get_batch_individual(self):
        """
        batch_size only support 1 for each GPU
        :return:
        """
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)
        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(self.parfetch(iroidb))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

    def parfetch(self, iroidb):
        # get testing data for multigpu

        imgs, hms, labels, roidb = self.get_video_label(iroidb, self.cfg)

        #im_info = np.array([roidb[0]['im_info']], dtype=np.float32)
        data = {'data': imgs} #, 'heatmap': hms}
        label = {'label1': [labels for i in range(4)],'label2':[labels]}

        return {'data': data, 'label': label}

    def transform(self, im, pixel_means):
        """
        transform into mxnet tensor
        substract pixel size and transform to correct format
        :param im: [height, width, channel] in BGR
        :param pixel_means: [B, G, R pixel means]
        :return: [batch, channel, height, width]
        """
        im_tensor = np.zeros((3, im.shape[0], im.shape[1]))
        for i in range(3):
            im_tensor[i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
        return im_tensor

    def transform_norm(self, im):
        """
        transform into mxnet tensor
        substract pixel size and transform to correct format
        :param im: [height, width, channel] in BGR
        :param pixel_means: [B, G, R pixel means]
        :return: [batch, channel, height, width]
        """
        im_tensor = np.zeros((1, im.shape[0], im.shape[1]))
        im_tensor[0, :, :] = im[:, :] / 255.0

        return im_tensor

    def get_images_label(self, iroidb, config):
        """
        preprocess image and return processed roidb
        :param roidb: a list of roidb
        :return: list of img as in mxnet format
        roidb add new item['im_info']
        0 --- x (width, second dim of im)
        |
        y (height, first dim of im)
        """
        num_images = len(iroidb)
        processed_ims = []
        processed_labels = []
        processed_roidb = []
        for i in range(num_images):
            roi_rec = iroidb[i]

            assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
            im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

            if iroidb[i]['flipped']:
                im = im[:, ::-1, :]

            new_rec = roi_rec.copy()
            scale_ind = random.randrange(len(config.SCALES))
            target_size = config.SCALES[scale_ind][0]
            max_size = config.SCALES[scale_ind][1]

            im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
            im_tensor = self.transform(im, config.network.PIXEL_MEANS)
            processed_ims.append(im_tensor)
            processed_labels.append(str(int(roi_rec['label'])-1))

            im_info = [im_tensor.shape[1], im_tensor.shape[2], im_scale]
            new_rec['im_info'] = im_info
            processed_roidb.append(new_rec)

        return processed_ims, processed_labels, processed_roidb


    def TemporalCenterCrop(self, frame_indices, duration):
        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (duration // 2))
        end_index = min(begin_index + duration, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= duration:
                break
            out.append(index)

        return out


    def TemporalRandomCrop(self, frame_indices, duration):
        rand_end = max(0, len(frame_indices) - duration - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + duration, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= duration:
                break
            out.append(index)

        return out

    def get_video_label(self, iroidb, config):
        """
        preprocess all frames and their hearmap for one video and return processed roidb
        :param roidb: a list of roidb
        :return: list of img as in mxnet format
        roidb add new item['im_info']
        0 --- x (width, second dim of im)
        |
        y (height, first dim of im)
        """
        n_batch = len(iroidb)
        assert(n_batch == 1)
        iroidb = iroidb[0]
        num_images = len(iroidb['images'])
        processed_ims = []
        processed_hms = []
        processed_labels = int(iroidb['label'])-1
        roi_rec = iroidb
        processed_roidb = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im_info = []

        n_samples_for_each_video = config.n_samples_for_each_video
        assert(n_samples_for_each_video == 1)
        sample_duration = config.sample_duration
        frame_indices = list(range(0, num_images))
        frame_indices = self.TemporalRandomCrop(frame_indices, sample_duration)

        for i in frame_indices:

            assert os.path.exists(roi_rec['images'][i]), '%s does not exist'.format(roi_rec['images'][i])
            im = cv2.imread(roi_rec['images'][i], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if iroidb['flipped']:
                im = im[:, ::-1, :]
            im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
            im_tensor = self.transform(im, config.network.PIXEL_MEANS)
            processed_ims.append(im_tensor)
            im_info.append([im_tensor.shape[1], im_tensor.shape[2], im_scale])

            # heat-map loading
            #target_size = 15
            #max_size = 20
            heatmap_path = roi_rec['images'][i].replace('JPG', 'HeatMap')
            assert os.path.exists(heatmap_path), '%s does not exist'.format(heatmap_path)
            im_heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
            if iroidb['flipped']:
                im = im[:, ::-1, :]
            im_heatmap, im_heatmap_scale = resize(im_heatmap, 15, 20, stride=config.network.IMAGE_STRIDE)
            im_heatmap_tensor = self.transform_norm(im_heatmap)
            processed_hms.append(im_heatmap_tensor)

        processed_roidb['im_info'] = im_info

        return processed_ims, processed_hms, processed_labels, processed_roidb

