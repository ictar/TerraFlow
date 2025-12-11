/* Compiler & Export Logic */

function findInput(nodeId, portName) {
    const conn = state.connections.find(c => c.targetNode === nodeId && c.targetPort === portName);
    if (!conn) return null;
    return state.nodes.find(n => n.id === conn.sourceNode);
}

function findAllInputs(nodeId, portName) {
    const conns = state.connections.filter(c => c.targetNode === nodeId && c.targetPort === portName);
    return conns.map(conn => state.nodes.find(n => n.id === conn.sourceNode)).filter(n => n);
}

function validateDetails(trainerNode) {
    if (!trainerNode) throw new Error("Graph must contain a Trainer node.");
    
    const taskNode = findInput(trainerNode.id, 'task');
    if (!taskNode) throw new Error("Trainer must have a Task connected.");

    const dataNode = findInput(trainerNode.id, 'datamodule');
    if (!dataNode) throw new Error("Trainer must have a DataModule connected.");
    
    return { taskNode, dataNode };
}

// Strategy 1: Hydra Config (For Notebook Export)
function buildHydraConfig() {
    const trainerNode = state.nodes.find(n => n.type === 'TrainerConfig');
    const { taskNode, dataNode } = validateDetails(trainerNode);
    const modelNode = findInput(taskNode.id, 'model_args');
    const optimNode = findInput(taskNode.id, 'optim_args');

    const bands = (dataNode.data.bands || 'BLUE,GREEN,RED,NIR').split(',');
    
    // Construct Object
    const fullConfig = {
        defaults: [
             { "/terratorch/datamodules": "generic_non_geo_segmentation" }, // Simplified default for Hydra
             { "/terratorch/tasks": taskNode.type },
             { "/terratorch/trainers": "default_trainer" } // Using simple default for Hydra base
        ],
        trainer: {
            max_epochs: parseInt(trainerNode.data.max_epochs),
            accelerator: trainerNode.data.accelerator,
            precision: trainerNode.data.precision,
            devices: trainerNode.data.devices === 'auto' ? 'auto' : (parseInt(trainerNode.data.devices) || trainerNode.data.devices),
            strategy: trainerNode.data.strategy,
            num_nodes: parseInt(trainerNode.data.num_nodes),
            check_val_every_n_epoch: parseInt(trainerNode.data.check_val_every_n_epoch),
            log_every_n_steps: parseInt(trainerNode.data.log_every_n_steps)
        },
        datamodule: {
            _target_: `terratorch.datamodules.${dataNode.data.class_path}`,
            batch_size: parseInt(dataNode.data.batch_size),
            num_workers: parseInt(dataNode.data.num_workers),
            bands: bands
        },
        task: {
            _target_: `terratorch.tasks.${taskNode.type}`,
            loss: taskNode.data.loss,
            ignore_index: parseInt(taskNode.data.ignore_index),
            model_args: {
                decoder: "UperNetDecoder",
                backbone: modelNode ? modelNode.data.backbone : "prithvi_eo_v1_100",
                num_classes: parseInt(taskNode.data.num_classes),
                backbone_pretrained: modelNode ? (modelNode.data.pretrained === 'True') : true
            }
        }
    };

    if (optimNode) {
        fullConfig.task.optimizer = {
             _target_: `torch.optim.${optimNode.data.type}`,
             lr: parseFloat(optimNode.data.lr),
             weight_decay: parseFloat(optimNode.data.weight_decay)
        };
    }
    
    return { yaml: jsyaml.dump(fullConfig), lossType: taskNode.data.loss };
}

// Strategy 2: Lightning CLI Config (For YAML Export)
function buildCLIConfig() {
    const trainerNode = state.nodes.find(n => n.type === 'TrainerConfig');
    const { taskNode, dataNode } = validateDetails(trainerNode);
    const modelNode = findInput(taskNode.id, 'model_args');
    const optimNode = findInput(taskNode.id, 'optim_args');
    
    // New: Find connected Logger and Callback
    const loggerNode = findInput(trainerNode.id, 'logger');
    const callbackNodes = findAllInputs(trainerNode.id, 'callbacks');

    const bands = (dataNode.data.bands || 'BLUE,GREEN,RED,NIR').split(',');
    const backbone = modelNode ? modelNode.data.backbone : 'prithvi_eo_v1_100';
    
    // Logic for Necks based on Backbone (Reference from User request)
    // Logic for Necks based on Backbone (Reference from User request)
    let necks = [];
    if (backbone.includes('prithvi')) {
        let indices = [2, 5, 8, 11]; // Default for 100m / prithvi_eo_v1_100
        if (backbone.includes('300')) indices = [5, 11, 17, 23]; // prithvi_eo_v2_300
        if (backbone.includes('600')) indices = [7, 15, 23, 31];
        
        necks = [
            { name: "SelectIndices", indices: indices },
            { name: "ReshapeTokensToImage" }
        ];
    } else if (backbone.includes('clay') || backbone.includes('satmae') || backbone.includes('scale_mae')) {
        // ViT-based models often need reshaping if they output tokens
        // For now, leaving empty or adding specific necks if known. 
        // Clay v1 usually works with a specific decoder adapter or neck.
        // Assuming default behavior or adding a placeholder comment for expansion.
        // necks = [{ name: "ReshapeTokensToImage" }]; // Example if needed
    }
    
    // Construct Callbacks List
    const callbacks = [];
    
    // If user hasn't added any callbacks, maybe default to RichProgressBar? 
    // Or just always add it if not present? 
    // Let's iterate connected nodes.
    
    if (callbackNodes.length === 0) {
        // Default behavior if nothing connected
        callbacks.push({ class_path: "RichProgressBar" });
        callbacks.push({ class_path: "LearningRateMonitor", init_args: { logging_interval: "epoch" } });
    } else {
        callbackNodes.forEach(node => {
            if (node.type === 'EarlyStopping') {
                callbacks.push({
                    class_path: "lightning.pytorch.callbacks.EarlyStopping",
                    init_args: {
                        patience: parseInt(node.data.patience),
                        monitor: node.data.monitor,
                        mode: node.data.mode
                    }
                });
            } else if (node.type === 'ModelCheckpoint') {
                callbacks.push({
                    class_path: "lightning.pytorch.callbacks.ModelCheckpoint",
                    init_args: {
                        monitor: node.data.monitor,
                        mode: node.data.mode,
                        save_top_k: parseInt(node.data.save_top_k),
                        filename: node.data.filename
                    }
                });
            } else if (node.type === 'LearningRateMonitor') {
                callbacks.push({
                    class_path: "LearningRateMonitor",
                    init_args: { logging_interval: node.data.logging_interval }
                });
            } else if (node.type === 'RichProgressBar') {
                callbacks.push({ class_path: "RichProgressBar" });
            }
        });
    }

    // Define loggerConfig
    let loggerConfig = null;
    if (loggerNode) {
        const type = loggerNode.data.type;
        const args = { save_dir: loggerNode.data.save_dir, name: loggerNode.data.name };
        
        let classPath = "lightning.pytorch.loggers.TensorBoardLogger"; // Default
        if (type === 'CSVLogger') {
            classPath = "lightning.pytorch.loggers.CSVLogger";
        } else if (type === 'WandbLogger') {
            classPath = "lightning.pytorch.loggers.WandbLogger";
            // Add project, entity if available
            if (loggerNode.data.project) args.project = loggerNode.data.project;
            if (loggerNode.data.entity) args.entity = loggerNode.data.entity;
        }
        loggerConfig = { class_path: classPath, init_args: args };
    }

    // Define callbackConfigs
    const callbackConfigs = callbacks.length > 0 ? callbacks : null;

    // --- 6. Construct CLI Config ---
    const taskClass = taskNode.type === 'PixelwiseRegressionTask' 
        ? 'terratorch.tasks.PixelwiseRegressionTask'
        : 'terratorch.tasks.SemanticSegmentationTask';

        // --- 5. Tiled Inference ---
        let tiledNode = null;
        if (taskNode) {
            tiledNode = findAllInputs(taskNode.id, 'tiled_inference')[0];
        }

        // --- 6. Model factory Inputs (Backbone, Decoder, Head, Neck) ---
        let backboneNode = null, decoderNode = null, headNode = null, neckNode = null;
        if (modelNode) {
            backboneNode = findAllInputs(modelNode.id, 'backbone')[0];
            decoderNode = findAllInputs(modelNode.id, 'decoder')[0];
            headNode = findAllInputs(modelNode.id, 'head')[0];
            neckNode = findAllInputs(modelNode.id, 'neck')[0];
        }

        // Helper: Construct Necks
        let necksConfig = [];
        if (neckNode) {
            // User connected a Neck node (Explicit Config)
            const indices = neckNode.data.indices.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
            necksConfig.push({ name: 'SelectIndices', indices: indices });
            if (neckNode.data.reshape === 'True') {
                necksConfig.push({ name: 'ReshapeTokensToImage' });
            }
        } else {
            // Default behavior if no Neck connected but using Prithvi
            necksConfig = getNecksForBackbone(backboneNode ? backboneNode.data.model_name : 'prithvi_eo_v1_100');
        }

        // --- 7. LR Scheduler ---
        const schedulerNode = findAllInputs(taskNode.id, 'scheduler_args')[0];
        let schedulerConfig = null;
        if (schedulerNode) {
            if (schedulerNode.data.type === 'ReduceLROnPlateau') {
                schedulerConfig = {
                    class_path: 'ReduceLROnPlateau',
                    init_args: {
                        monitor: schedulerNode.data.monitor,
                        patience: parseInt(schedulerNode.data.patience),
                        mode: 'min' 
                    }
                };
            } else {
                schedulerConfig = {
                     class_path: 'CosineAnnealingLR',
                     init_args: {
                         T_max: parseInt(schedulerNode.data.t_max)
                     }
                };
            }
        } else {
            // Fallback default
            schedulerConfig = {
                class_path: "CosineAnnealingLR",
                init_args: {
                    T_max: parseInt(trainerNode.data.max_epochs)
                }
            };
        }

        const cliConfig = {
            experiment_name: state.globalConfig.experiment_name || 'my_experiment',
            seed_everything: state.globalConfig.seed_everything,
            trainer: {
                accelerator: trainerNode.data.accelerator,
                strategy: trainerNode.data.strategy,
                devices: trainerNode.data.devices === 'auto' ? 'auto' : (parseInt(trainerNode.data.devices) || trainerNode.data.devices),
                num_nodes: parseInt(trainerNode.data.num_nodes),
                precision: trainerNode.data.precision,
                logger: loggerConfig,
                callbacks: callbackConfigs,
                max_epochs: parseInt(trainerNode.data.max_epochs),
                check_val_every_n_epoch: parseInt(trainerNode.data.check_val_every_n_epoch),
                log_every_n_steps: parseInt(trainerNode.data.log_every_n_steps),
                enable_checkpointing: trainerNode.data.enable_checkpointing === 'True',
                default_root_dir: trainerNode.data.default_root_dir
            },
            data: {
                class_path: dataNode.data.class_path,
                init_args: {
                    data_root: dataNode.data.data_root,
                    batch_size: parseInt(dataNode.data.batch_size),
                    num_workers: parseInt(dataNode.data.num_workers),
                    constant_scale: parseFloat(dataNode.data.constant_scale),
                    no_data_replace: parseInt(dataNode.data.no_data_replace),
                    no_label_replace: parseInt(dataNode.data.no_label_replace),
                    use_metadata: dataNode.data.use_metadata === 'True',
                    bands: bands,
                    train_transform: gatherTransforms(dataNode.id, 'train_transform'),
                    val_transform: gatherTransforms(dataNode.id, 'val_transform'),
                    test_transform: gatherTransforms(dataNode.id, 'test_transform')
                }
            },
            model: {
                class_path: taskClass,
                init_args: {
                    model_args: {
                        // Backbone Args
                        backbone: backboneNode ? backboneNode.data.model_name : 'prithvi_eo_v1_100',
                        backbone_pretrained: backboneNode ? (backboneNode.data.pretrained === 'True') : true,
                        backbone_bands: bands, 
                        in_channels: backboneNode ? parseInt(backboneNode.data.in_channels) : 6,
                        num_frames: backboneNode ? parseInt(backboneNode.data.num_frames) : 1,
                        backbone_drop_path_rate: backboneNode ? parseFloat(backboneNode.data.drop_path_rate) : 0.0,
                        backbone_window_size: backboneNode ? parseInt(backboneNode.data.window_size) : 8,
                        
                        // Decoder Args
                        decoder: decoderNode ? decoderNode.data.decoder_name : 'UperNetDecoder',
                        decoder_channels: decoderNode ? parseInt(decoderNode.data.decoder_channels) : 256,
                        decoder_scale_modules: decoderNode ? (decoderNode.data.scale_modules === 'True') : true,
                        
                        // Head Args
                        head_dropout: headNode ? parseFloat(headNode.data.dropout) : 0.1,
                        head_learned_upscale_layers: headNode ? parseInt(headNode.data.learned_upscale_layers) : 1,
                        head_final_act: (headNode && headNode.data.final_act !== 'None') ? `torch.nn.${headNode.data.final_act}` : undefined,

                        // Task Specific + Necks
                        num_classes: taskNode.data.num_classes ? parseInt(taskNode.data.num_classes) : undefined,
                        rescale: true,
                        necks: necksConfig 
                    },
                    loss: taskNode.data.loss,
                    ignore_index: parseInt(taskNode.data.ignore_index),
                    freeze_backbone: false, 
                    freeze_decoder: false,
                    model_factory: taskNode.data.model_factory || 'EncoderDecoderFactory'
                }
            },
            optimizer: optimNode ? {
                class_path: optimNode.data.type === 'SGD' ? 'torch.optim.SGD' : 'torch.optim.AdamW',
                init_args: {
                    lr: parseFloat(optimNode.data.lr),
                    weight_decay: parseFloat(optimNode.data.weight_decay)
                }
            } : {},
            lr_scheduler: schedulerConfig
        };

    // Add Tiled Inference if connected
    if (tiledNode) {
        cliConfig.model.init_args.tiled_inference_parameters = {
            h_crop: parseInt(tiledNode.data.h_crop),
            w_crop: parseInt(tiledNode.data.w_crop),
            h_stride: parseInt(tiledNode.data.h_stride),
            w_stride: parseInt(tiledNode.data.w_stride),
            average_patches: tiledNode.data.average_patches === 'True'
        };
    }
    
    // Cleanup nulls
    if (!cliConfig.trainer.logger) delete cliConfig.trainer.logger;
    
    return { yaml: jsyaml.dump(cliConfig) };
}

// Helper to gather and sort transforms (by Y position for sequence)
function gatherTransforms(nodeId, portName) {
    const nodes = findAllInputs(nodeId, portName);
    // Sort by Y position (top to bottom execution)
    nodes.sort((a, b) => a.y - b.y);
    
    return nodes.map(node => {
        if (node.type === 'AlbumentationsResize') {
            return { 
                class_path: "albumentations.Resize", 
                init_args: { height: parseInt(node.data.height), width: parseInt(node.data.width) } 
            };
        } else if (node.type === 'AlbumentationsHorizontalFlip') {
            return { 
                class_path: "albumentations.HorizontalFlip", 
                init_args: { p: parseFloat(node.data.p) } 
            };
        } else if (node.type === 'AlbumentationsVerticalFlip') {
             return { 
                class_path: "albumentations.VerticalFlip", 
                init_args: { p: parseFloat(node.data.p) } 
            };
        } else if (node.type === 'ToTensorV2') {
             return { class_path: "ToTensorV2" }; // Usually albumentations.pytorch.ToTensorV2 but Terratorch alias? assume ToTensorV2 for now or needs full path?
             // Using alias as per user example
        }
        return null;
    }).filter(t => t);
}

// --- Helpers ---

function mdCell(source) { return { cell_type: "markdown", metadata: {}, source: [source] }; }
function codeCell(source) { return { cell_type: "code", execution_count: null, metadata: {}, outputs: [], source: source }; }

function generateNotebookJSON(yamlString, lossType) {
     const cells = [];
    cells.push(mdCell(`# 🌍 TerraFlow Job\n\nGenerated by TerraFlow UI. \n\n**Components:**\n- **Task:** Segmentation (${lossType})`));
    cells.push(codeCell(["!nvidia-smi"]));
    cells.push(codeCell([
        "%%capture\n",
        "!pip install -q terratorch torchgeo pytorch-lightning\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n"
    ]));
    cells.push(codeCell([
        "from omegaconf import OmegaConf\n",
        `config_str = """\n${yamlString}"""\n`,
        "cfg = OmegaConf.create(config_str)\n",
        "print('✅ Architecture Configuration Loaded')"
    ]));
    cells.push(codeCell([
        "from terratorch.cli_tools import build_datamodule\n",
        "import matplotlib.pyplot as plt\n\n",
        "print('🔍 Instantiating DataModule...')\n",
        "dm = build_datamodule(cfg.datamodule)\n",
        "dm.setup('fit')\n",
        "sample = next(iter(dm.train_dataloader()))\n",
        "print(f'📦 Batch Shape: {sample[\"image\"].shape}')"
    ]));
    cells.push(codeCell([
        "import pytorch_lightning as pl\n",
        "from terratorch.cli_tools import build_lightning_module\n\n",
        "print('🧠 Instantiating Task & Model...')\n",
        "task = build_lightning_module(cfg.task)\n\n",
        "trainer = pl.Trainer(**cfg.trainer)\n",
        "print('✅ Ready to Train!')"
    ]));
    cells.push(codeCell([
        "# trainer.fit(task, dm)"
    ]));
    
    return {
        metadata: { kernelspec: { display_name: "Python 3", language: "python", name: "python3" } },
        nbformat: 4,
        nbformat_minor: 4,
        cells: cells
    };
}

function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// --- Main Export Functions ---

function exportNotebook() {
    try {
        const { yaml, lossType } = buildHydraConfig();
        const notebook = generateNotebookJSON(yaml, lossType);
        downloadJSON(notebook, 'TerraFlow_Job.ipynb');
    } catch (e) {
        alert("⚠️ Export Failed: " + e.message);
    }
}

function exportYaml() {
     try {
        const { yaml } = buildCLIConfig();
        // Create blob and download as .yaml
        const blob = new Blob([yaml], { type: 'text/yaml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'config.yaml';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    } catch (e) {
        alert("⚠️ Export Failed: " + e.message);
    }
}
