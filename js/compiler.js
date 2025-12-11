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

    const cliConfig = {
        seed_everything: state.globalConfig.seed_everything,
        trainer: {
            accelerator: trainerNode.data.accelerator,
            strategy: trainerNode.data.strategy,
            devices: trainerNode.data.devices === 'auto' ? 'auto' : (parseInt(trainerNode.data.devices) || trainerNode.data.devices),
            num_nodes: parseInt(trainerNode.data.num_nodes),
            precision: trainerNode.data.precision,
            precision: trainerNode.data.precision,
            logger: (() => {
                if (!loggerNode) return null;
                const type = loggerNode.data.type;
                const args = { save_dir: loggerNode.data.save_dir, name: loggerNode.data.name };
                
                let classPath = "lightning.pytorch.loggers.TensorBoardLogger";
                if (type === 'Wandb') {
                    classPath = "lightning.pytorch.loggers.WandbLogger";
                    args.project = loggerNode.data.project;
                    args.log_model = true;
                } else if (type === 'CSV') {
                    classPath = "lightning.pytorch.loggers.CSVLogger";
                } else if (type === 'MLFlow') {
                    classPath = "lightning.pytorch.loggers.MLFlowLogger";
                    args.experiment_name = loggerNode.data.project;
                    args.run_name = loggerNode.data.name;
                    delete args.name; // MLFlow uses run_name
                }
                
                return {
                    class_path: classPath,
                    init_args: args
                };
            })(),
            callbacks: callbacks,
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
                // Common args usually needed for these datamodules
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
            class_path: `terratorch.tasks.${taskNode.type.replace('SegmentationTask', 'SemanticSegmentationTask')}`,
            init_args: {
                model_args: {
                    backbone_pretrained: modelNode ? (modelNode.data.pretrained === 'True') : true,
                    backbone: backbone,
                    decoder: "UperNetDecoder",
                    decoder_channels: 256,
                    decoder_scale_modules: true,
                    num_classes: parseInt(taskNode.data.num_classes),
                    rescale: true,
                    backbone_bands: bands,
                    head_dropout: 0.1,
                    necks: necks
                },
                loss: taskNode.data.loss,
                ignore_index: parseInt(taskNode.data.ignore_index),
                freeze_backbone: false,
                freeze_decoder: false,
                model_factory: "EncoderDecoderFactory"
            }
        },
        optimizer: optimNode ? {
            class_path: `torch.optim.${optimNode.data.type}`,
            init_args: {
                lr: parseFloat(optimNode.data.lr),
                weight_decay: parseFloat(optimNode.data.weight_decay)
            }
        } : {},
        lr_scheduler: {
            class_path: "CosineAnnealingLR",
            init_args: { T_max: parseInt(trainerNode.data.max_epochs) }
        }
    };
    
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
