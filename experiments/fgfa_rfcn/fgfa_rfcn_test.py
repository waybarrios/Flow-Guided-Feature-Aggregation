# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'fgfa_rfcn'))

import test_ucf101

if __name__ == "__main__":
    """
    ctx_idx: gpu index
    sample_idx_start: the start index of samples for testing
    sample_idx_end: the end index of samples for testing
    """
    test_ucf101.main(ctx_idx=0, sample_idx_start=0, sample_idx_end=1000)
