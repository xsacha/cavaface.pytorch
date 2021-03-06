import torch

configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results
        
        DATA_ROOT = '/home/imagus/datasets/', # the parent root where your train/val/test data are stored
        RECORD_DIR = '/home/imagus/datasets/record.txt', # the dataset record dir
        VAL_DATA_ROOT = '/home/imagus/datasets/data', # the parent root where your val/test data are stored
        MODEL_ROOT = 'models', # the root to buffer your checkpoints
        LOG_ROOT = 'log', # the root to log your train/val status
        IS_RESUME = True,
        BACKBONE_RESUME_ROOT = "models/Backbone_AttentionNet_IR_92_Epoch_18_Time_2020-08-09-08-31_checkpoint.pth",
        HEAD_RESUME_ROOT = "models/Head_CurricularFace_Epoch_18_Time_2020-08-09-08-31_checkpoint.pth",
        
        BACKBONE_NAME = 'AttentionNet_IR_92', # support: ['MobileFaceNet', 'ResNet_50', 'ResNet_101', 'ResNet_152', 
                                #'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152',
                                #'AttentionNet_IR_56', 'AttentionNet_IRSE_56','AttentionNet_IR_92', 'AttentionNet_IRSE_92']
        HEAD_NAME = "CurricularFace", # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax', 'ArcNegFace', 'CurricularFace', 'SVX']
        LOSS_NAME = 'Softmax', # support: [''Softmax', Focal', 'HardMining', 'LabelSmooth']
        
        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 224,
        EVAL_FREQ = 8000, #for ms1m, batch size 1024, EVAL_FREQ=2000
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        
        LR = 0.0001, # initial LR
        LR_SCHEDULER = 'cosine', # step/multi_step/cosine
        WARMUP_EPOCH = 0, 
        WARMUP_LR = 0.0,
        START_EPOCH = 0, #start epoch
        NUM_EPOCH = 55, # total epoch number
        LR_STEP_SIZE = 10, # 'step' scheduler, period of learning rate decay. 
        LR_DECAY_EPOCH = [10, 18, 22], # ms1m epoch stages to decay learning rate
        LR_DECAT_GAMMA = 0.1, # multiplicative factor of learning rate decay
        LR_END = 1e-5, # minimum learning rate
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        
        WORLD_SIZE = 1,
        RANK = 0,
        GPU = [0,1,2], # specify your GPU ids
        DIST_BACKEND = 'nccl', # 'nccl', 'gloo'
        DIST_URL = 'tcp://localhost:23456',
        NUM_WORKERS = 1,
        TEST_GPU_ID = [0,1,2],

        # Data Augmentation
        RANDAUGMENT = False,
        RANDAUGMENT_N = 2, # random pick numer of aug typr form aug_list 
        RANDAUGMENT_M = 9,
        RANDOM_ERASING = False,
        MIXUP = False,
        MIXUP_ALPHA = 1.0,
        MIXUP_PROB = 0.5,
        CUTOUT = False, 
        CUTMIX = False, 
        CUTMIX_ALPHA = 1.0,
        CUTMIX_PROB = 0.5
    ),
}
