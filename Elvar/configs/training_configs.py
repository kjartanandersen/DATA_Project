TRAINING_CONFIGS = {
    'epochs_20': {
        'num_epochs': 20,
        'batch_size': 128,
        'optimizer': 'SGD',
        'optimizer_params': {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0001
        },
        'scheduler': 'MultiStepLR',
        'scheduler_params': {
            'milestones': [5, 10, 15],
            'gamma': 0.1
        }
    },
    'epochs_50': {
        'num_epochs': 50,
        'batch_size': 128,
        'optimizer': 'SGD',
        'optimizer_params': {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0001
        },
        'scheduler': 'MultiStepLR',
        'scheduler_params': {
            'milestones': [15, 30, 40], 
            'gamma': 0.1
        }
    },
    'epochs_100': {
        'num_epochs': 100,
        'batch_size': 128,
        'optimizer': 'SGD',
        'optimizer_params': {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0001
        },
        'cheduler': 'MultiStepLR',
        'cheduler_params': {
            'milestones': [30, 60, 80],
            'gamma': 0.1
        }
    }
}