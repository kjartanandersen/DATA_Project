from models.resnet import ResNet

MODELS = {
    'resnet20': {
        'model_class': ResNet,
        'model_params': {
            'n': 3, # number of blocks in each stack (3 for ResNet20)
            'shortcuts': True
        }
    },'esnet32': {
        'odel_class': 'ResNet',
        'odel_params': {
            'n': 5,  # number of blocks in each stack (5 for ResNet32)
            'hortcuts': True
        }
    },
    'esnet44': {
        'odel_class': 'ResNet',
        'odel_params': {
            'n': 7,  # number of blocks in each stack (7 for ResNet44)
            'hortcuts': True
        }
    },
    
}