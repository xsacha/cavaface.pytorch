import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn.functional as F
from config import configurations
from backbone.resnet import *
from backbone.resnet_irse import *
from backbone.mobilefacenet import *
from backbone.resattnet import *
from backbone.efficientnet import *
from backbone.resnest import *
from head.metrics import *
from loss.loss import FocalLoss
from util.utils import get_val_data, perform_val, get_time, AverageMeter, accuracy
from tqdm import tqdm
import os
import time
import numpy as np
import scipy
import pickle

if __name__ == '__main__':

    #======= hyperparameters & data loaders =======#
    cfg = configurations[1]
    torch.backends.cudnn.benchmark = True
    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    VAL_DATA_ROOT = cfg['VAL_DATA_ROOT'] # the parent root where your train/val/test data are stored
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    INPUT_SIZE = cfg['INPUT_SIZE']
    BATCH_SIZE = cfg['BATCH_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    GPU_ID = cfg['TEST_GPU_ID'] # specify your GPU ids
    print("Overall Configurations:")
    print(cfg)
    #val_data_dir = os.path.join(VAL_DATA_ROOT, 'val_data')
    val_data_dir = VAL_DATA_ROOT
    lfw, nist, multiracial, challenging, muct, lfw_issame, nist_issame, multiracial_issame, challenging_issame, muct_issame = get_val_data(VAL_DATA_ROOT)

    #======= model =======#
    BACKBONE_DICT = {'MobileFaceNet': MobileFaceNet,
                     'ResNet_50': ResNet_50, 'ResNet_101': ResNet_101, 'ResNet_152': ResNet_152,
                     'IR_50': IR_50, 'IR_100': IR_100, 'IR_101': IR_101, 'IR_152': IR_152, 'IR_185': IR_185, 'IR_200': IR_200,
                     'IR_SE_50': IR_SE_50, 'IR_SE_100': IR_SE_100, 'IR_SE_101': IR_SE_101, 'IR_SE_152': IR_SE_152, 'IR_SE_185': IR_SE_185, 'IR_SE_200': IR_SE_200,
                     'EfficientNet': efficientnet,
                     'AttentionNet_IR_56': AttentionNet_IR_56,'AttentionNet_IRSE_56': AttentionNet_IRSE_56,'AttentionNet_IR_92': AttentionNet_IR_92,'AttentionNet_IRSE_92': AttentionNet_IRSE_92,
                     'ResNeSt_50': resnest50, 'ResNeSt_101': resnest101, 'ResNeSt_100': resnest100
                    } 

    BACKBONE = BACKBONE_DICT[BACKBONE_NAME](INPUT_SIZE)
    print("=" * 60)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)
    if BACKBONE_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}'".format(BACKBONE_RESUME_ROOT))
            exit()
        print("=" * 60)
    torch.quantization.fuse_modules(BACKBONE, [['input_layer.0', 'input_layer.1'],
                                          ['residual_block1.res_layer.3', 'residual_block1.res_layer.4'],
                                          ['attention_module1.share_residual_block.res_layer.3', 'attention_module1.share_residual_block.res_layer.4'],
                                          ['attention_module1.trunk_branches.0.res_layer.3',     'attention_module1.trunk_branches.0.res_layer.4'],
                                          ['attention_module1.trunk_branches.1.res_layer.3',     'attention_module1.trunk_branches.1.res_layer.4'],
                                          ['attention_module1.mask_block1.res_layer.3',          'attention_module1.mask_block1.res_layer.4'],
                                          ['attention_module1.skip_connect1.res_layer.3',        'attention_module1.skip_connect1.res_layer.4'],
                                          ['attention_module1.mask_block2.res_layer.3',          'attention_module1.mask_block2.res_layer.4'],
                                          ['attention_module1.skip_connect2.res_layer.3',        'attention_module1.skip_connect2.res_layer.4'],
                                          ['attention_module1.mask_block3.0.res_layer.3',        'attention_module1.mask_block3.0.res_layer.4'],
                                          ['attention_module1.mask_block3.1.res_layer.3',        'attention_module1.mask_block3.1.res_layer.4'],
                                          ['attention_module1.mask_block4.res_layer.3',          'attention_module1.mask_block4.res_layer.4'],
                                          ['attention_module1.mask_block5.res_layer.3',          'attention_module1.mask_block5.res_layer.4'],
                                          #['attention_module1.mask_block6.0',                    'attention_module1.mask_block6.1'],
                                          ['attention_module1.mask_block6.2',                    'attention_module1.mask_block6.3',                    'attention_module1.mask_block6.4'],
                                          ['attention_module1.last_block.res_layer.3',           'attention_module1.last_block.res_layer.4'],
                                          #['residual_block2.shortcut_layer.0', 'residual_block2.shortcut_layer.1'],
                                          ['residual_block2.res_layer.3', 'residual_block2.res_layer.4'],
                                          ['attention_module2.first_residual_blocks.res_layer.3','attention_module2.first_residual_blocks.res_layer.4'],
                                          ['attention_module2.trunk_branches.0.res_layer.3',     'attention_module2.trunk_branches.0.res_layer.4'],
                                          ['attention_module2.trunk_branches.1.res_layer.3',     'attention_module2.trunk_branches.1.res_layer.4'],
                                          ['attention_module2.softmax1_blocks.res_layer.3',      'attention_module2.softmax1_blocks.res_layer.4'],
                                          ['attention_module2.skip1_connection_residual_block.res_layer.3','attention_module2.skip1_connection_residual_block.res_layer.4'],
                                          ['attention_module2.softmax2_blocks.0.res_layer.3',    'attention_module2.softmax2_blocks.0.res_layer.4'],
                                          ['attention_module2.softmax2_blocks.1.res_layer.3',    'attention_module2.softmax2_blocks.1.res_layer.4'],
                                          ['attention_module2.softmax3_blocks.res_layer.3',      'attention_module2.softmax3_blocks.res_layer.4'],
                                          #['attention_module2.softmax4_blocks.0',                'attention_module2.softmax4_blocks.1'],
                                          ['attention_module2.softmax4_blocks.2',                'attention_module2.softmax4_blocks.3',                'attention_module2.softmax4_blocks.4'],
                                          ['attention_module2.last_blocks.res_layer.3',          'attention_module2.last_blocks.res_layer.4'],
                                          #['residual_block3.shortcut_layer.0', 'residual_block3.shortcut_layer.1'],
                                          ['residual_block3.res_layer.3', 'residual_block3.res_layer.4'],
                                          ['attention_module3.first_residual_blocks.res_layer.3','attention_module3.first_residual_blocks.res_layer.4'],
                                          ['attention_module3.trunk_branches.0.res_layer.3',     'attention_module3.trunk_branches.0.res_layer.4'],
                                          ['attention_module3.trunk_branches.1.res_layer.3',     'attention_module3.trunk_branches.1.res_layer.4'],
                                          ['attention_module3.softmax1_blocks.0.res_layer.3',    'attention_module3.softmax1_blocks.0.res_layer.4'],
                                          ['attention_module3.softmax1_blocks.1.res_layer.3',    'attention_module3.softmax1_blocks.1.res_layer.4'],
                                          #['attention_module3.softmax2_blocks.0',                'attention_module3.softmax2_blocks.1'],
                                          ['attention_module3.softmax2_blocks.2',                'attention_module3.softmax2_blocks.3',      'attention_module3.softmax2_blocks.4'],
                                          ['attention_module3.last_blocks.res_layer.3',          'attention_module3.last_blocks.res_layer.4'],
                                          ['residual_block4.shortcut_layer.0', 'residual_block4.shortcut_layer.1'],
                                          ['residual_block4.res_layer.3', 'residual_block4.res_layer.4'],
                                          ['residual_block5.res_layer.3', 'residual_block5.res_layer.4'],
                                          ['residual_block6.res_layer.3', 'residual_block6.res_layer.4'],
                                         ], inplace=True)
    BACKBONE.cuda()

    if len(GPU_ID) > 1:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)

    print("Perform Evaluation on LFW, NIST, Challenging, Multiracial,...")
    start = time.time()
    accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(EMBEDDING_SIZE, BATCH_SIZE, traced, lfw, lfw_issame)
    print("Accuracy: {}".format(accuracy_lfw))
    torch.cuda.synchronize()
    print(time.time() - start)
    accuracy_nist, best_threshold_nist, roc_curve_nist = perform_val(EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, nist, nist_issame)
    print("Challenging...")
    accuracy_challenging, best_threshold_challenging, roc_curve_challenging = perform_val(EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, challenging, challenging_issame)
    print("Multiracial...")
    accuracy_multiracial, best_threshold_multiracial, roc_curve_multiracial = perform_val(EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, multiracial, multiracial_issame)

    print("Evaluation: LFW Acc: {}, NIST Acc: {}, Challenging Acc: {}, Multiracial Acc: {}".format(accuracy_lfw, accuracy_nist, accuracy_challenging, accuracy_multiracial))
