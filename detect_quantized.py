from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from config import configurations
from backbone.resnet import *
from backbone.resnet_irse import *
from backbone.mobilefacenet import *
from backbone.resattnet import *
from backbone.efficientpolyface import *
from backbone.resnest import *
from backbone.efficientnet import *
import torch.autograd.profiler as profiler

from torch._C import MobileOptimizerType

from torch.utils import mkldnn

import time
import torch.quantization
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.utils import mkldnn
#from torch.cuda.amp import autocast

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='Backbone_AttentionNet_IR_92_Epoch_27_Time_2021-06-26-21-38_checkpoint.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--cl', action="store_true", default=False, help='Use channels last')
parser.add_argument('--mkl', action="store_true", default=False, help='Use mkldnn inference')
parser.add_argument('--amp', action="store_true", default=False, help='Use AMP inference')
parser.add_argument('--mobile', action="store_true", default=False, help='Optimise for mobile')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def test(model, device, args):
    resize = 1

    # testing begin
    for i in range(100):
        image_path = "gbush.jpg"
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_raw = cv2.resize(img_raw, (112, 112))

        img = np.float32(img_raw)

        img -= 127.5
        img /= 127.5
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        if args.cl:
            img = img.to(memory_format=torch.channels_last)
        if args.amp:
            img = img.half()
        if args.mkl:
            img = img.to_mkldnn()

        tic = time.time()
        #with torch.cpu.amp.autocast(args.mkl):
        with profiler.profile(with_stack=True, profile_memory=False) as prof:
            feature = model(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))
        print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    device = torch.device("cpu" if args.cpu else "cuda")
    # net and model
    net = AttentionNet_IR_92([112, 112])
    net = load_model(net, args.trained_model, args.cpu)
    net = net.to(device)
    net.eval()
    print('Finished loading model!')
    #print(net)
    cudnn.benchmark = True
    print(net)
    # insert observers
    #torch.quantization.prepare(net, inplace=True)
    # Calibrate the model and collect statistics
    #test(net, device, args)

    # Effnet
    '''
    torch.quantization.fuse_modules(net, [['features.init_block.conv.conv', 'features.init_block.conv.bn'],
                                          ['features.stage1.unit1.dw_conv.conv','features.stage1.unit1.dw_conv.bn'],
                                          ['features.stage1.unit1.se.conv1','features.stage1.unit1.se.activ'],
                                          ['features.stage1.unit1.pw_conv.conv','features.stage1.unit1.pw_conv.bn'],
                                          ['features.stage1.unit2.dw_conv.conv','features.stage1.unit2.dw_conv.bn'],
                                          ['features.stage1.unit2.se.conv1','features.stage1.unit2.se.activ'],
                                          ['features.stage1.unit2.pw_conv.conv','features.stage1.unit2.pw_conv.bn'],
                                          ['features.stage2.unit1.conv1.conv','features.stage2.unit1.conv1.bn'],
                                          ['features.stage2.unit1.conv2.conv','features.stage2.unit1.conv2.bn'],
                                          ['features.stage2.unit1.se.conv1','features.stage2.unit1.se.activ'],
                                          ['features.stage2.unit1.conv3.conv','features.stage2.unit1.conv3.bn'],
                                          ['features.stage2.unit2.conv1.conv','features.stage2.unit2.conv1.bn'],
                                          ['features.stage2.unit2.conv2.conv','features.stage2.unit2.conv2.bn'],
                                          ['features.stage2.unit2.se.conv1','features.stage2.unit2.se.activ'],
                                          ['features.stage2.unit2.conv3.conv','features.stage2.unit2.conv3.bn'],
                                          ['features.stage2.unit3.conv1.conv','features.stage2.unit3.conv1.bn'],
                                          ['features.stage2.unit3.conv2.conv','features.stage2.unit3.conv2.bn'],
                                          ['features.stage2.unit3.se.conv1','features.stage2.unit3.se.activ'],
                                          ['features.stage2.unit3.conv3.conv','features.stage2.unit3.conv3.bn'],
                                          ['features.stage2.unit4.conv1.conv','features.stage2.unit4.conv1.bn'],
                                          ['features.stage2.unit4.conv2.conv','features.stage2.unit4.conv2.bn'],
                                          ['features.stage2.unit4.se.conv1','features.stage2.unit4.se.activ'],
                                          ['features.stage2.unit4.conv3.conv','features.stage2.unit4.conv3.bn'],
                                          ['features.stage3.unit1.conv1.conv','features.stage3.unit1.conv1.bn'],
                                          ['features.stage3.unit1.conv2.conv','features.stage3.unit1.conv2.bn'],
                                          ['features.stage3.unit1.se.conv1','features.stage3.unit1.se.activ'],
                                          ['features.stage3.unit1.conv3.conv','features.stage3.unit1.conv3.bn'],
                                          ['features.stage3.unit2.conv1.conv','features.stage3.unit2.conv1.bn'],
                                          ['features.stage3.unit2.conv2.conv','features.stage3.unit2.conv2.bn'],
                                          ['features.stage3.unit2.se.conv1','features.stage3.unit2.se.activ'],
                                          ['features.stage3.unit2.conv3.conv','features.stage3.unit2.conv3.bn'],
                                          ['features.stage3.unit3.conv1.conv','features.stage3.unit3.conv1.bn'],
                                          ['features.stage3.unit3.conv2.conv','features.stage3.unit3.conv2.bn'],
                                          ['features.stage3.unit3.se.conv1','features.stage3.unit3.se.activ'],
                                          ['features.stage3.unit3.conv3.conv','features.stage3.unit3.conv3.bn'],
                                          ['features.stage3.unit4.conv1.conv','features.stage3.unit4.conv1.bn'],
                                          ['features.stage3.unit4.conv2.conv','features.stage3.unit4.conv2.bn'],
                                          ['features.stage3.unit4.se.conv1','features.stage3.unit4.se.activ'],
                                          ['features.stage3.unit4.conv3.conv','features.stage3.unit4.conv3.bn'],
                                          ['features.stage4.unit1.conv1.conv','features.stage4.unit1.conv1.bn'],
                                          ['features.stage4.unit1.conv2.conv','features.stage4.unit1.conv2.bn'],
                                          ['features.stage4.unit1.se.conv1','features.stage4.unit1.se.activ'],
                                          ['features.stage4.unit1.conv3.conv','features.stage4.unit1.conv3.bn'],
                                          ['features.stage4.unit2.conv1.conv','features.stage4.unit2.conv1.bn'],
                                          ['features.stage4.unit2.conv2.conv','features.stage4.unit2.conv2.bn'],
                                          ['features.stage4.unit2.se.conv1','features.stage4.unit2.se.activ'],
                                          ['features.stage4.unit2.conv3.conv','features.stage4.unit2.conv3.bn'],
                                          ['features.stage4.unit3.conv1.conv','features.stage4.unit3.conv1.bn'],
                                          ['features.stage4.unit3.conv2.conv','features.stage4.unit3.conv2.bn'],
                                          ['features.stage4.unit3.se.conv1','features.stage4.unit3.se.activ'],
                                          ['features.stage4.unit3.conv3.conv','features.stage4.unit3.conv3.bn'],
                                          ['features.stage4.unit4.conv1.conv','features.stage4.unit4.conv1.bn'],
                                          ['features.stage4.unit4.conv2.conv','features.stage4.unit4.conv2.bn'],
                                          ['features.stage4.unit4.se.conv1',  'features.stage4.unit4.se.activ'],
                                          ['features.stage4.unit4.conv3.conv','features.stage4.unit4.conv3.bn'],
                                          ['features.stage4.unit5.conv1.conv','features.stage4.unit5.conv1.bn'],
                                          ['features.stage4.unit5.conv2.conv','features.stage4.unit5.conv2.bn'],
                                          ['features.stage4.unit5.se.conv1',  'features.stage4.unit5.se.activ'],
                                          ['features.stage4.unit5.conv3.conv','features.stage4.unit5.conv3.bn'],
                                          ['features.stage4.unit6.conv1.conv','features.stage4.unit6.conv1.bn'],
                                          ['features.stage4.unit6.conv2.conv','features.stage4.unit6.conv2.bn'],
                                          ['features.stage4.unit6.se.conv1',  'features.stage4.unit6.se.activ'],
                                          ['features.stage4.unit6.conv3.conv','features.stage4.unit6.conv3.bn'],
                                          ['features.stage4.unit7.conv1.conv','features.stage4.unit7.conv1.bn'],
                                          ['features.stage4.unit7.conv2.conv','features.stage4.unit7.conv2.bn'],
                                          ['features.stage4.unit7.se.conv1',  'features.stage4.unit7.se.activ'],
                                          ['features.stage4.unit7.conv3.conv','features.stage4.unit7.conv3.bn'],
                                          ['features.stage4.unit8.conv1.conv','features.stage4.unit8.conv1.bn'],
                                          ['features.stage4.unit8.conv2.conv','features.stage4.unit8.conv2.bn'],
                                          ['features.stage4.unit8.se.conv1',  'features.stage4.unit8.se.activ'],
                                          ['features.stage4.unit8.conv3.conv','features.stage4.unit8.conv3.bn'],
                                          ['features.stage4.unit9.conv1.conv','features.stage4.unit9.conv1.bn'],
                                          ['features.stage4.unit9.conv2.conv','features.stage4.unit9.conv2.bn'],
                                          ['features.stage4.unit9.se.conv1',  'features.stage4.unit9.se.activ'],
                                          ['features.stage4.unit9.conv3.conv','features.stage4.unit9.conv3.bn'],
                                          ['features.stage4.unit10.conv1.conv','features.stage4.unit10.conv1.bn'],
                                          ['features.stage4.unit10.conv2.conv','features.stage4.unit10.conv2.bn'],
                                          ['features.stage4.unit10.se.conv1',  'features.stage4.unit10.se.activ'],
                                          ['features.stage4.unit10.conv3.conv','features.stage4.unit10.conv3.bn'],
                                          ['features.stage4.unit11.conv1.conv','features.stage4.unit11.conv1.bn'],
                                          ['features.stage4.unit11.conv2.conv','features.stage4.unit11.conv2.bn'],
                                          ['features.stage4.unit11.se.conv1',  'features.stage4.unit11.se.activ'],
                                          ['features.stage4.unit11.conv3.conv','features.stage4.unit11.conv3.bn'],
                                          ['features.stage4.unit12.conv1.conv','features.stage4.unit12.conv1.bn'],
                                          ['features.stage4.unit12.conv2.conv','features.stage4.unit12.conv2.bn'],
                                          ['features.stage4.unit12.se.conv1',  'features.stage4.unit12.se.activ'],
                                          ['features.stage4.unit12.conv3.conv','features.stage4.unit12.conv3.bn'],
                                          ['features.stage5.unit1.conv1.conv','features.stage5.unit1.conv1.bn'],
                                          ['features.stage5.unit1.conv2.conv','features.stage5.unit1.conv2.bn'],
                                          ['features.stage5.unit1.se.conv1','features.stage5.unit1.se.activ'],
                                          ['features.stage5.unit1.conv3.conv','features.stage5.unit1.conv3.bn'],
                                          ['features.stage5.unit2.conv1.conv','features.stage5.unit2.conv1.bn'],
                                          ['features.stage5.unit2.conv2.conv','features.stage5.unit2.conv2.bn'],
                                          ['features.stage5.unit2.se.conv1','features.stage5.unit2.se.activ'],
                                          ['features.stage5.unit2.conv3.conv','features.stage5.unit2.conv3.bn'],
                                          ['features.stage5.unit3.conv1.conv','features.stage5.unit3.conv1.bn'],
                                          ['features.stage5.unit3.conv2.conv','features.stage5.unit3.conv2.bn'],
                                          ['features.stage5.unit3.se.conv1','features.stage5.unit3.se.activ'],
                                          ['features.stage5.unit3.conv3.conv','features.stage5.unit3.conv3.bn'],
                                          ['features.stage5.unit4.conv1.conv','features.stage5.unit4.conv1.bn'],
                                          ['features.stage5.unit4.conv2.conv','features.stage5.unit4.conv2.bn'],
                                          ['features.stage5.unit4.se.conv1',  'features.stage5.unit4.se.activ'],
                                          ['features.stage5.unit4.conv3.conv','features.stage5.unit4.conv3.bn'],
                                          ['features.stage5.unit5.conv1.conv','features.stage5.unit5.conv1.bn'],
                                          ['features.stage5.unit5.conv2.conv','features.stage5.unit5.conv2.bn'],
                                          ['features.stage5.unit5.se.conv1',  'features.stage5.unit5.se.activ'],
                                          ['features.stage5.unit5.conv3.conv','features.stage5.unit5.conv3.bn'],
                                          ['features.stage5.unit6.conv1.conv','features.stage5.unit6.conv1.bn'],
                                          ['features.stage5.unit6.conv2.conv','features.stage5.unit6.conv2.bn'],
                                          ['features.stage5.unit6.se.conv1',  'features.stage5.unit6.se.activ'],
                                          ['features.stage5.unit6.conv3.conv','features.stage5.unit6.conv3.bn'],
                                          ['features.stage5.unit7.conv1.conv','features.stage5.unit7.conv1.bn'],
                                          ['features.stage5.unit7.conv2.conv','features.stage5.unit7.conv2.bn'],
                                          ['features.stage5.unit7.se.conv1',  'features.stage5.unit7.se.activ'],
                                          ['features.stage5.unit7.conv3.conv','features.stage5.unit7.conv3.bn'],
                                          ['features.stage5.unit8.conv1.conv','features.stage5.unit8.conv1.bn'],
                                          ['features.stage5.unit8.conv2.conv','features.stage5.unit8.conv2.bn'],
                                          ['features.stage5.unit8.se.conv1',  'features.stage5.unit8.se.activ'],
                                          ['features.stage5.unit8.conv3.conv','features.stage5.unit8.conv3.bn'],
                                          ['features.stage5.unit9.conv1.conv','features.stage5.unit9.conv1.bn'],
                                          ['features.stage5.unit9.conv2.conv','features.stage5.unit9.conv2.bn'],
                                          ['features.stage5.unit9.se.conv1',  'features.stage5.unit9.se.activ'],
                                          ['features.stage5.unit9.conv3.conv','features.stage5.unit9.conv3.bn'],
                                          ['features.stage5.unit10.conv1.conv','features.stage5.unit10.conv1.bn'],
                                          ['features.stage5.unit10.conv2.conv','features.stage5.unit10.conv2.bn'],
                                          ['features.stage5.unit10.se.conv1',  'features.stage5.unit10.se.activ'],
                                          ['features.stage5.unit10.conv3.conv','features.stage5.unit10.conv3.bn'],
                                          ['features.final_block.conv', 'features.final_block.bn'],
                                          ['output.conv_6_dw.conv','output.conv_6_dw.bn'],
                                         ], inplace=True)
    '''
    # V7
    '''
    torch.quantization.fuse_modules(net, [['input_layer.0', 'input_layer.1'],
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
                                          ['attention_module2_2.first_residual_blocks.res_layer.3', 'attention_module2_2.first_residual_blocks.res_layer.4'],
                                          ['attention_module2_2.trunk_branches.0.res_layer.3', 'attention_module2_2.trunk_branches.0.res_layer.4'],
                                          ['attention_module2_2.trunk_branches.1.res_layer.3', 'attention_module2_2.trunk_branches.1.res_layer.4'],
                                          ['attention_module2_2.softmax1_blocks.res_layer.3',      'attention_module2_2.softmax1_blocks.res_layer.4'],
                                          ['attention_module2_2.skip1_connection_residual_block.res_layer.3','attention_module2_2.skip1_connection_residual_block.res_layer.4'],
                                          ['attention_module2_2.softmax2_blocks.0.res_layer.3',    'attention_module2_2.softmax2_blocks.0.res_layer.4'],
                                          ['attention_module2_2.softmax2_blocks.1.res_layer.3',    'attention_module2_2.softmax2_blocks.1.res_layer.4'],
                                          ['attention_module2_2.softmax3_blocks.res_layer.3',      'attention_module2_2.softmax3_blocks.res_layer.4'],
                                          ['attention_module2_2.softmax4_blocks.2',                'attention_module2_2.softmax4_blocks.3',                'attention_module2_2.softmax4_blocks.4'],
                                          ['attention_module2_2.last_blocks.res_layer.3',          'attention_module2_2.last_blocks.res_layer.4'],
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
    '''

    print(net)


    # convert to quantized version
    #torch.quantization.convert(net, inplace=True)
    #net = net.to(device)
    #net = mkldnn.to_mkldnn(net)
    inp = torch.randn(1, 3, 112, 112).to(device)
    if args.amp:
        inp = inp.half()
        net = net.half()
    if args.cl:
        inp = inp.to(memory_format=torch.channels_last)
        net = net.to(memory_format=torch.channels_last)
    if args.mkl:
        inp = inp.to_mkldnn()
        #net = mkldnn.to_mkldnn(net)
    #with torch.cpu.amp.autocast(args.mkl):
    #net = torch.jit.script(net)
    net = torch.jit.trace(net, inp, check_trace=False)
    net = torch.jit.freeze(net)
    #net = torch.jit.optimize_for_inference(net)
    
    #torchnet = optimize_for_mobile(net, {MobileOptimizerType.INSERT_FOLD_PREPACK_OPS})
    if args.mobile:
        torchnet = optimize_for_mobile(net)
        torchnet.save('v8.pt')
        test(torchnet, device, args)
    else:
        net.save('v8mask.pt')
        #net2 = torch.jit.load('v8.pt')
        test(net, device, args)


