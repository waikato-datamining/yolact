from data.config import dataset_base, yolact_plus_resnet50_config

external_dataset = dataset_base.copy({
    'name': 'External Dataset',

    # "/data/images" refers to the directory inside the docker container
    'train_images': '/data/images',
    'train_info':   '/data/images/train.json',

    'valid_images': '/data/images',
    'valid_info':   '/data/images/test.json',

    'has_gt': True,
    'class_names': ('class1', 'class2', 'class3')
})


external_config = yolact_plus_resnet50_config.copy({
    'name': 'External config',  # this name gets used for storing model files: NAME_XXX_YYY.pth
    
    # Dataset stuff
    'dataset': external_dataset,  # references the above dataset via its variable name
    'num_classes': 4,  # labels + 1 for background

    'max_iter': 120000,
    'lr_steps': (60000, 100000),
})

