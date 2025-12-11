/* Node Definitions (TerraFlow Schema) */
const NODE_TYPES = {
    // [DATA MODULE] - Emerald Theme
    'HLSDataModule': {
        title: 'HLS Data Module',
        color: '#10b981', // Emerald 500
        inputs: [],
        outputs: ['datamodule'], 
        params: [
            { key: 'batch_size', label: 'BATCH SIZE', type: 'number', default: 4 },
            { key: 'num_workers', label: 'WORKERS', type: 'number', default: 4 },
            { key: 'bands', label: 'BANDS', type: 'text', default: 'BLUE,GREEN,RED,NIR' }
        ]
    },
    // [MODEL FACTORY] - Violet Theme
    'ModelFactory': {
        title: 'Model Factory',
        color: '#8b5cf6', // Violet 500
        inputs: [],
        outputs: ['model_args'], 
        params: [
            { key: 'backbone', label: 'BACKBONE', type: 'select', options: ['prithvi_100m', 'prithvi_300m', 'swin_base'], default: 'prithvi_100m' },
            { key: 'pretrained', label: 'PRETRAINED', type: 'select', options: ['True', 'False'], default: 'True' }
        ]
    },
    // [OPTIMIZER] - Amber Theme
    'OptimizerConfig': {
        title: 'Optimizer',
        color: '#f59e0b', // Amber 500
        inputs: [],
        outputs: ['optim_args'], 
        params: [
            { key: 'lr', label: 'LEARNING RATE', type: 'text', default: '1e-4' },
            { key: 'weight_decay', label: 'WEIGHT DECAY', type: 'text', default: '0.05' },
            { key: 'type', label: 'OPTIMIZER TYPE', type: 'select', options: ['AdamW', 'SGD'], default: 'AdamW' }
        ]
    },
    // [TASK] - Orange Theme
    'SegmentationTask': {
        title: 'Segmentation Task',
        color: '#f97316', // Orange 500
        inputs: ['model_args', 'optim_args'], 
        outputs: ['task'], 
        params: [
            { key: 'loss', label: 'LOSS FUNCTION', type: 'select', options: ['ce', 'dice', 'focal'], default: 'ce' },
            { key: 'num_classes', label: 'NUM CLASSES', type: 'number', default: 2 },
            { key: 'ignore_index', label: 'IGNORE INDEX', type: 'number', default: -1 }
        ]
    },
    // [LOGGER] - Pink Theme
    'TensorBoardLogger': {
        title: 'TensorBoard Logger',
        color: '#ec4899', // Pink 500
        inputs: [],
        outputs: ['logger'],
        params: [
            { key: 'name', label: 'NAME', type: 'text', default: 'terraflow_run' },
            { key: 'save_dir', label: 'SAVE DIR', type: 'text', default: 'logs' }
        ]
    },
    // [CALLBACK] - Rose Theme
    'EarlyStopping': {
        title: 'Early Stopping',
        color: '#f43f5e', // Rose 500
        inputs: [],
        outputs: ['callback'],
        params: [
            { key: 'patience', label: 'PATIENCE', type: 'number', default: 20 },
            { key: 'monitor', label: 'MONITOR', type: 'text', default: 'val/loss' }
        ]
    },
    // [TRAINER] - Blue Theme
    'TrainerConfig': {
        title: 'Lightning Trainer',
        color: '#3b82f6', // Blue 500
        inputs: ['task', 'datamodule', 'logger', 'early_stopping'],
        outputs: [],
        params: [
            { key: 'max_epochs', label: 'MAX EPOCHS', type: 'number', default: 50 },
            { key: 'accelerator', label: 'ACCELERATOR', type: 'select', options: ['gpu', 'cpu', 'auto'], default: 'auto' },
            { key: 'devices', label: 'DEVICES', type: 'text', default: 'auto' },
            { key: 'strategy', label: 'STRATEGY', type: 'text', default: 'auto' },
            { key: 'num_nodes', label: 'NUM NODES', type: 'number', default: 1 },
            { key: 'precision', label: 'PRECISION', type: 'select', options: ['16-mixed', '32'], default: '16-mixed' },
            { key: 'check_val_every_n_epoch', label: 'VAL CHECK INTERVAL', type: 'number', default: 2 },
            { key: 'log_every_n_steps', label: 'LOG STEP INTERVAL', type: 'number', default: 10 }
        ]
    }
};
