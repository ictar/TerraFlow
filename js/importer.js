/* Import Logic from YAML */

function importYaml() {
    document.getElementById('import-file').click();
}

function handleFileImport(input) {
    const file = input.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const content = e.target.result;
            const config = jsyaml.load(content);
            reconstructGraph(config);
            input.value = ''; // Reset
        } catch (e) {
            console.error(e);
            alert("❌ Import Failed: Invalid YAML file.\n" + e.message);
        }
    };
    reader.readAsText(file);
}

function reconstructGraph(config, silent = false) {
    if (!silent && !confirm("Importing will clear the current canvas. Continue?")) return;
    
    // 1. Clear Canvas
    state.nodes = [];
    state.connections = [];
    nodesContainer.innerHTML = ''; 
    nodesContainer.appendChild(svgLayer); // CRITICAL FIX: Restore SVG layer that was wiped out
    svgLayer.innerHTML = '';

    try {
        // 2. Parse & Create Nodes using initialData
        
        // --- Trainer ---
        const trainerData = {};
        if (config.trainer) {
            trainerData.max_epochs = config.trainer.max_epochs;
            trainerData.accelerator = config.trainer.accelerator;
            trainerData.precision = config.trainer.precision;
            trainerData.strategy = config.trainer.strategy;
            trainerData.devices = config.trainer.devices;
            trainerData.num_nodes = config.trainer.num_nodes;
            trainerData.check_val_every_n_epoch = config.trainer.check_val_every_n_epoch;
            trainerData.log_every_n_steps = config.trainer.log_every_n_steps;
            trainerData.enable_checkpointing = config.trainer.enable_checkpointing === false ? 'False' : 'True';
            trainerData.default_root_dir = config.trainer.default_root_dir || 'checkpoints';
            state.globalConfig.seed_everything = config.seed_everything !== undefined ? config.seed_everything : 0;
            // Update UI Panel
            if (typeof updateGlobalUI === 'function') updateGlobalUI();
        }
        const trainerNode = addNode('TrainerConfig', 900, 380, trainerData);
            
        // --- Logger & Callbacks ---
        let loggerNode, esNode;
        if (config.trainer) {
            // Parse Logger
            const loggerConfig = config.trainer.logger;
            if (loggerConfig) {
                const init = loggerConfig.init_args || {};
                let type = 'TensorBoard';
                let project = 'terraflow_project';
                let name = init.name || 'terraflow_run';
                
                const cls = loggerConfig.class_path || '';
                if (cls.includes('Wandb')) {
                    type = 'Wandb';
                    project = init.project;
                } else if (cls.includes('CSV')) {
                    type = 'CSV';
                } else if (cls.includes('MLFlow')) {
                    type = 'MLFlow';
                    project = init.experiment_name;
                    name = init.run_name;
                }

                const loggerData = {
                    type: type,
                    project: project || 'terraflow_project',
                    name: name,
                    save_dir: init.save_dir || 'logs'
                };
                loggerNode = addNode('Logger', 900, 200, loggerData);
            }
            
            // Parse Callbacks
            const callbackNodesData = [];
            if (Array.isArray(config.trainer.callbacks)) {
                config.trainer.callbacks.forEach((cb, idx) => {
                    const cp = cb.class_path || '';
                    if (cp.includes('EarlyStopping')) {
                         const d = { 
                             patience: cb.init_args?.patience, 
                             monitor: cb.init_args?.monitor || 'val/loss',
                             mode: cb.init_args?.mode || 'min'
                         };
                         callbackNodesData.push({ type: 'EarlyStopping', data: d });
                    } else if (cp.includes('ModelCheckpoint')) {
                         const d = {
                             monitor: cb.init_args?.monitor || 'val/loss',
                             mode: cb.init_args?.mode || 'min',
                             save_top_k: cb.init_args?.save_top_k || 1,
                             filename: cb.init_args?.filename
                         };
                         callbackNodesData.push({ type: 'ModelCheckpoint', data: d });
                    } else if (cp.includes('LearningRateMonitor')) {
                         const d = { logging_interval: cb.init_args?.logging_interval || 'epoch' };
                         callbackNodesData.push({ type: 'LearningRateMonitor', data: d });
                    } else if (cp.includes('RichProgressBar')) {
                         callbackNodesData.push({ type: 'RichProgressBar', data: {} });
                    }
                });
            }
            
            // Create Callback Nodes
            // We need to store them to connect later
            var createdCallbackNodes = [];
            callbackNodesData.forEach((item, i) => {
                const node = addNode(item.type, 700 + (i*50), 600 + (i*150), item.data);
                createdCallbackNodes.push(node);
            });
        }

        // --- Data ---
        let dataNode;
        if (config.data) {
            const init = config.data.init_args || {};
            const dataData = {
                class_path: config.data.class_path || 'GenericNonGeoSegmentationDataModule',
                data_root: init.data_root || 'data/dataset',
                batch_size: init.batch_size,
                num_workers: init.num_workers,
                bands: Array.isArray(init.bands) ? init.bands.join(',') : init.bands,
                constant_scale: init.constant_scale,
                no_data_replace: init.no_data_replace,
                no_label_replace: init.no_label_replace,
                use_metadata: init.use_metadata ? 'True' : 'False'
            };
            dataNode = addNode('DataModule', 500, 520, dataData);
            
            // Parse Transforms
            ['train_transform', 'val_transform', 'test_transform'].forEach((key, keyIdx) => {
                if (Array.isArray(init[key])) {
                    init[key].forEach((t, i) => {
                         let type = null;
                         let tData = {};
                         
                         if (t.class_path.includes('Resize')) {
                             type = 'AlbumentationsResize';
                             tData = { height: t.init_args.height, width: t.init_args.width };
                         } else if (t.class_path.includes('HorizontalFlip')) {
                             type = 'AlbumentationsHorizontalFlip';
                             tData = { p: t.init_args.p };
                         } else if (t.class_path.includes('VerticalFlip')) {
                             type = 'AlbumentationsVerticalFlip';
                             tData = { p: t.init_args.p };
                         } else if (t.class_path.includes('ToTensorV2')) {
                             type = 'ToTensorV2';
                         }
                         
                         if (type) {
                             const tNode = addNode(type, 150 + (i * 180), 600 + (keyIdx * 150), tData);
                             // Connect
                             state.connections.push({
                                 sourceNode: tNode.id, sourcePort: 'transform',
                                 targetNode: dataNode.id, targetPort: key
                             });
                         }
                    });
                }
            });
        }

        // --- Model / Task ---
        let taskNode;
        let modelFactoryNode;
        if (config.model) {
            const init = config.model.init_args || {};
            const taskData = {
                 loss: init.loss,
                 ignore_index: init.ignore_index,
                 num_classes: (init.model_args || {}).num_classes
            };
            taskNode = addNode('SegmentationTask', 500, 220, taskData);
            
            if (init.model_args) {
                const modelData = {
                    backbone: init.model_args.backbone,
                    pretrained: init.model_args.backbone_pretrained === false ? 'False' : 'True'
                };
                modelFactoryNode = addNode('ModelFactory', 100, 150, modelData);
            }
        }

        // --- Optimizer ---
        let optimNode;
        if (config.optimizer) {
            const init = config.optimizer.init_args || {};
            const type = config.optimizer.class_path ? config.optimizer.class_path.split('.').pop() : 'AdamW';
            const optimData = {
                lr: init.lr,
                weight_decay: init.weight_decay,
                type: ['AdamW', 'SGD'].includes(type) ? type : 'AdamW'
            };
            optimNode = addNode('OptimizerConfig', 100, 380, optimData);
        }

        // 3. Reconnect
        // Force a layout update if needed (browsers usually handle this on getBoundingClientRect)
        
        if (modelFactoryNode && taskNode) {
            state.connections.push({ sourceNode: modelFactoryNode.id, sourcePort: 'model_args', targetNode: taskNode.id, targetPort: 'model_args' });
        }
        if (optimNode && taskNode) {
            state.connections.push({ sourceNode: optimNode.id, sourcePort: 'optim_args', targetNode: taskNode.id, targetPort: 'optim_args' });
        }
        if (taskNode && trainerNode) {
            state.connections.push({ sourceNode: taskNode.id, sourcePort: 'task', targetNode: trainerNode.id, targetPort: 'task' });
        }
        if (dataNode && trainerNode) {
            state.connections.push({ sourceNode: dataNode.id, sourcePort: 'datamodule', targetNode: trainerNode.id, targetPort: 'datamodule' });
        }
        if (loggerNode && trainerNode) {
             state.connections.push({ sourceNode: loggerNode.id, sourcePort: 'logger', targetNode: trainerNode.id, targetPort: 'logger' });
        }
        if (createdCallbackNodes.length > 0 && trainerNode) {
             createdCallbackNodes.forEach(cbNode => {
                 state.connections.push({ sourceNode: cbNode.id, sourcePort: 'callback', targetNode: trainerNode.id, targetPort: 'callbacks' });
             });
        }
        
        // Use multiple render strategies to ensure lines appear
        console.log("Rebuilding connections...", state.connections.length);
        renderConnections();
        
        // Failsafe for slower DOM updates
        setTimeout(() => {
            renderConnections();
            if (typeof autoLayout === 'function') autoLayout();
        }, 100);

    } catch (err) {
        console.error(err);
        alert("⚠️ Error parsing structure: " + err.message);
    }
}
