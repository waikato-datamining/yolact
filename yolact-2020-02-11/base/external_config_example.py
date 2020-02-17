from data.config import dataset_base, yolact_resnet50_config             

external_dataset = dataset_base.copy({
    'name': 'External Dataset',

    'train_images': '/data/images',
    'train_info':   '/data/images/train.json',

    'valid_images': '/data/images',
    'valid_info':   '/data/images/test.json',

    'has_gt': True,
    'class_names': ('class1', 'class2', 'class3')
})


external_config = yolact_resnet50_config.copy({
    'name': 'External config',
    
    # Dataset stuff
    'dataset': external_dataset,
    'num_classes': 4,  # labels + 1 for background

    'max_iter': 120000,
    'lr_steps': (60000, 100000),
    
    'backbone': yolact_resnet50_config.backbone.copy({
        'pred_scales': [[32], [64], [128], [256], [512]],
        'use_square_anchors': False,
    })
})

