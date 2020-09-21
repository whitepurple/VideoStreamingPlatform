class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 100
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = './data/Data/lfw/Dataset/'  # train path
    train_list = './data/Data/lfw/Kface_rand_train.txt' # train list
    val_list = './data/Data/train.txt'

    test_root = '/data/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    eval_root = './data/ML_Data/Data_val'  # val or test path
    eval_list = './data/ML_Data/ML_val_list.txt' # val or test list

    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet18.pth'
    eval_model_path = 'checkpoints/resnet_18_KFace.pth'
    save_interval = 10

    train_batch_size = 8  # batch size
    test_batch_size = 1

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True
    gpu_id = '0'
    num_workers = -1
    print_freq = 100

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-1
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4
