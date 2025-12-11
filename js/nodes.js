/* Node Definitions (TerraFlow Schema) */
const NODE_TYPES = {
    // [DATA MODULE] - Emerald Theme
    // [DATA MODULE] - Emerald Theme
    // [DATA MODULE] - Emerald Theme
    'DataModule': {
        title: 'Data Module',
        color: '#10b981', // Emerald 500
        inputs: ['train_transform', 'val_transform', 'test_transform'],
        outputs: ['datamodule'], 
        params: [
            { 
                key: 'class_path', 
                label: 'CLASS PATH', 
                type: 'select', 
                options: [
                    'GenericNonGeoPixelwiseRegressionDataModule',
                    'GenericNonGeoSegmentationDataModule',
                    'Sen1Floods11NonGeoDataModule',
                    'Landsat7DataModule',
                    'TorchGeoDataModule'
                ], 
                default: 'GenericNonGeoSegmentationDataModule' 
            },
            { key: 'data_root', label: 'DATA ROOT', type: 'text', default: 'data/dataset' },
            { key: 'batch_size', label: 'BATCH SIZE', type: 'number', default: 4 },
            { key: 'num_workers', label: 'WORKERS', type: 'number', default: 4 },
            { key: 'bands', label: 'BANDS', type: 'text', default: 'BLUE,GREEN,RED,NIR' },
            { key: 'constant_scale', label: 'CONST SCALE', type: 'text', default: '0.0001' },
            { key: 'no_data_replace', label: 'NO DATA VAL', type: 'number', default: 0 },
            { key: 'no_label_replace', label: 'NO LABEL VAL', type: 'number', default: -1 },
            { key: 'use_metadata', label: 'USE METADATA', type: 'select', options: ['True', 'False'], default: 'False' }
        ]
    },
    // [TRANSFORMS] - Cyan Theme
    'AlbumentationsResize': {
        title: 'Resize',
        color: '#06b6d4', // Cyan 500
        inputs: [], outputs: ['transform'],
        params: [
            { key: 'height', label: 'HEIGHT', type: 'number', default: 224 },
            { key: 'width', label: 'WIDTH', type: 'number', default: 224 }
        ]
    },
    'AlbumentationsHorizontalFlip': {
        title: 'Horizontal Flip',
        color: '#06b6d4',
        inputs: [], outputs: ['transform'],
        params: [
            { key: 'p', label: 'PROBABILITY', type: 'text', default: '0.5' }
        ]
    },
    'AlbumentationsVerticalFlip': {
        title: 'Vertical Flip',
        color: '#06b6d4',
        inputs: [], outputs: ['transform'],
        params: [
            { key: 'p', label: 'PROBABILITY', type: 'text', default: '0.5' }
        ]
    },
    'ToTensorV2': {
        title: 'To Tensor',
        color: '#06b6d4',
        inputs: [], outputs: ['transform'],
        params: []
    },
    // [ARCHITECTURE] - Violet Theme
    'ModelBackbone': {
        title: 'Backbone',
        color: '#7c3aed', // Violet 600
        inputs: [],
        outputs: ['backbone_config'],
        params: [
            { 
                key: 'model_name', 
                label: 'MODEL', 
                type: 'select', 
                options: [
                    'prithvi_eo_v1_100', 
                    'prithvi_eo_v2_300', 
                    'prithvi_eo_v2_600',
                    'clay_v1_base', 
                    'satmae_vit_base_patch16', 
                    'scale_mae', 
                    'swin_base_patch4_window12_384', 
                    'resnet50', 
                    'dofa_base_patch16_224',
                    'terramind_v1_base'
                ], 
                default: 'prithvi_eo_v1_100' 
            },
            { key: 'pretrained', label: 'PRETRAINED', type: 'select', options: ['True', 'False'], default: 'True' },
            { key: 'in_channels', label: 'IN CHANNELS', type: 'number', default: 6 },
            { key: 'num_frames', label: 'NUM FRAMES', type: 'number', default: 1 },
            { key: 'drop_path_rate', label: 'DROP PATH RATE', type: 'text', default: '0.3' },
            { key: 'window_size', label: 'WINDOW SIZE', type: 'number', default: 8 }
        ]
    },
    'ModelDecoder': {
        title: 'Decoder',
        color: '#8b5cf6', // Violet 500
        inputs: [],
        outputs: ['decoder_config'],
        params: [
            { key: 'decoder_name', label: 'DECODER', type: 'select', options: ['UperNetDecoder', 'FCNDecoder', 'IdentityDecoder'], default: 'UperNetDecoder' },
            { key: 'decoder_channels', label: 'CHANNELS', type: 'number', default: 256 },
            { key: 'scale_modules', label: 'SCALE MODULES', type: 'select', options: ['True', 'False'], default: 'True' }
        ]
    },
    'ModelHead': {
        title: 'Task Head',
        color: '#a78bfa', // Violet 400
        inputs: [],
        outputs: ['head_config'],
        params: [
            { key: 'dropout', label: 'DROPOUT', type: 'text', default: '0.1' },
            { key: 'learned_upscale_layers', label: 'UPSCALE LAYERS', type: 'number', default: 1 },
            { key: 'final_act', label: 'FINAL ACTIVATION', type: 'select', options: ['None', 'ReLU', 'Sigmoid', 'Tanh'], default: 'None' }
        ]
    },
    'ModelNeck': {
        title: 'Model Neck',
        color: '#8b5cf6', // Violet 500
        inputs: [],
        outputs: ['neck_config'],
        params: [
            { key: 'indices', label: 'INDICES (comma-sep)', type: 'text', default: '2,5,8,11' },
            { key: 'reshape', label: 'RESHAPE TOKENS', type: 'select', options: ['True', 'False'], default: 'True' }
        ]
    },
    'ModelFactory': {
        title: 'Model Factory',
        color: '#5b21b6', // Violet 800 (Darker main node)
        inputs: ['backbone', 'decoder', 'head', 'neck'],
        outputs: ['model_args'], 
        params: [
             // Keeps simple default behavior if nothing connected, or simple params?
             // To enforce modularity, let's keep params empty or minimal.
             // But for backward compatibility/simplicity, maybe provide defaults? 
             // Let's assume input-driven for now.
        ]
    },
    // [TILED INFERENCE] - Violet Theme (Darker)
    'TiledInference': {
        title: 'Tiled Inference',
        color: '#6d28d9', // Violet 700
        inputs: [],
        outputs: ['tiled_inference'],
        params: [
             { key: 'h_crop', label: 'H CROP', type: 'number', default: 224 },
             { key: 'w_crop', label: 'W CROP', type: 'number', default: 224 },
             { key: 'h_stride', label: 'H STRIDE', type: 'number', default: 192 },
             { key: 'w_stride', label: 'W STRIDE', type: 'number', default: 192 },
             { key: 'average_patches', label: 'AVG PATCHES', type: 'select', options: ['True', 'False'], default: 'True' }
        ]
    },
    // [OPTIMIZER] - Amber Theme
    // [OPTIMIZER & SCHEDULER] - Amber Theme
    'OptimizerConfig': {
        title: 'Optimizer',
        color: '#f59e0b', // Amber 500
        inputs: [],
        outputs: ['optim_args'], 
        params: [
            { key: 'type', label: 'TYPE', type: 'select', options: ['AdamW', 'SGD'], default: 'AdamW' },
            { key: 'lr', label: 'LEARNING RATE', type: 'text', default: '1e-4' },
            { key: 'weight_decay', label: 'WEIGHT DECAY', type: 'text', default: '0.05' }
        ]
    },
    'LRScheduler': {
        title: 'LR Scheduler',
        color: '#d97706', // Amber 600
        inputs: [],
        outputs: ['scheduler_args'],
        params: [
            { key: 'type', label: 'TYPE', type: 'select', options: ['ReduceLROnPlateau', 'CosineAnnealingLR'], default: 'ReduceLROnPlateau' },
            { key: 'monitor', label: 'MONITOR (Plateau)', type: 'text', default: 'val/loss' },
            { key: 't_max', label: 'T_MAX (Cosine)', type: 'number', default: 50 },
            { key: 'patience', label: 'PATIENCE (Plateau)', type: 'number', default: 10 }
        ]
    },
    // [TASK] - Orange Theme
    // [TASK] - Orange Theme
    'SegmentationTask': {
        title: 'Segmentation Task',
        color: '#f97316', // Orange 500
        inputs: ['model_args', 'optim_args', 'scheduler_args', 'tiled_inference'],
        outputs: ['task'], 
        params: [
            { key: 'loss', label: 'LOSS FUNCTION', type: 'select', options: ['ce', 'dice', 'focal'], default: 'ce' },
            { key: 'num_classes', label: 'NUM CLASSES', type: 'number', default: 2 },
            { key: 'ignore_index', label: 'IGNORE INDEX', type: 'number', default: -1 },
            { key: 'model_factory', label: 'FACTORY', type: 'select', options: ['EncoderDecoderFactory', 'PrithviModelFactory'], default: 'EncoderDecoderFactory' }
        ]
    },
    'PixelwiseRegressionTask': {
        title: 'Regression Task',
        color: '#f97316', // Orange 500
        inputs: ['model_args', 'optim_args', 'scheduler_args', 'tiled_inference'], 
        outputs: ['task'], 
        params: [
            { key: 'loss', label: 'LOSS FUNCTION', type: 'select', options: ['rmse', 'mae', 'huber'], default: 'rmse' },
            { key: 'ignore_index', label: 'IGNORE INDEX', type: 'number', default: -1 },
            { key: 'model_factory', label: 'FACTORY', type: 'select', options: ['PrithviModelFactory', 'EncoderDecoderFactory'], default: 'PrithviModelFactory' }
        ]
    },
    // [LOGGER] - Pink Theme
    // [LOGGER] - Pink Theme
    'Logger': {
        title: 'Logger',
        color: '#ec4899', // Pink 500
        inputs: [],
        outputs: ['logger'],
        params: [
            { key: 'type', label: 'TYPE', type: 'select', options: ['TensorBoard', 'Wandb', 'CSV', 'MLFlow'], default: 'TensorBoard' },
            { key: 'project', label: 'PROJECT (Wandb/MLFlow)', type: 'text', default: 'terraflow_project' },
            { key: 'name', label: 'RUN NAME', type: 'text', default: 'run_version_1' },
            { key: 'save_dir', label: 'SAVE DIR', type: 'text', default: 'logs' }
        ]
    },
    // [CALLBACK] - Rose Theme
    // [CALLBACK] - Rose Theme
    'EarlyStopping': {
        title: 'Early Stopping',
        color: '#f43f5e', // Rose 500
        inputs: [],
        outputs: ['callback'],
        params: [
            { key: 'patience', label: 'PATIENCE', type: 'number', default: 20 },
            { key: 'monitor', label: 'MONITOR', type: 'text', default: 'val/loss' },
            { key: 'mode', label: 'MODE', type: 'select', options: ['min', 'max'], default: 'min' }
        ]
    },
    'ModelCheckpoint': {
        title: 'Model Checkpoint',
        color: '#f43f5e',
        inputs: [],
        outputs: ['callback'],
        params: [
            { key: 'monitor', label: 'MONITOR', type: 'text', default: 'val/loss' },
            { key: 'mode', label: 'MODE', type: 'select', options: ['min', 'max'], default: 'min' },
            { key: 'save_top_k', label: 'SAVE TOP K', type: 'number', default: 1 },
            { key: 'filename', label: 'FILENAME', type: 'text', default: '{epoch}-{val/loss:.2f}' }
        ]
    },
    'LearningRateMonitor': {
        title: 'LR Monitor',
        color: '#f43f5e',
        inputs: [],
        outputs: ['callback'],
        params: [
            { key: 'logging_interval', label: 'LOG INTERVAL', type: 'select', options: ['step', 'epoch'], default: 'epoch' }
        ]
    },
    'RichProgressBar': {
        title: 'Rich Progress Bar',
        color: '#f43f5e',
        inputs: [],
        outputs: ['callback'],
        params: []
    },
    // [TRAINER] - Blue Theme
    'TrainerConfig': {
        title: 'Lightning Trainer',
        color: '#3b82f6', // Blue 500
        inputs: ['task', 'datamodule', 'logger', 'callbacks'],
        outputs: [],
        params: [
            { key: 'max_epochs', label: 'MAX EPOCHS', type: 'number', default: 50 },
            { key: 'accelerator', label: 'ACCELERATOR', type: 'select', options: ['gpu', 'cpu', 'auto'], default: 'auto' },
            { key: 'devices', label: 'DEVICES', type: 'text', default: 'auto' },
            { key: 'strategy', label: 'STRATEGY', type: 'text', default: 'auto' },
            { key: 'num_nodes', label: 'NUM NODES', type: 'number', default: 1 },
            { key: 'precision', label: 'PRECISION', type: 'select', options: ['16-mixed', '32'], default: '16-mixed' },
            { key: 'check_val_every_n_epoch', label: 'VAL CHECK INTERVAL', type: 'number', default: 2 },
            { key: 'log_every_n_steps', label: 'LOG STEP INTERVAL', type: 'number', default: 10 },
            { key: 'enable_checkpointing', label: 'ENABLE CHECKPOINTING', type: 'select', options: ['True', 'False'], default: 'True' },
            { key: 'default_root_dir', label: 'ROOT DIR', type: 'text', default: 'checkpoints' }
        ]
    }
};
