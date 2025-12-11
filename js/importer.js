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

function reconstructGraph(config) {
    if (!confirm("Importing will clear the current canvas. Continue?")) return;
    
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
        }
        const trainerNode = addNode('TrainerConfig', 900, 380, trainerData);
            
        // --- Logger & Callbacks ---
        let loggerNode, esNode;
        if (config.trainer) {
            // Parse Logger
            if (config.trainer.logger && config.trainer.logger.init_args) {
                const loggerData = {
                    name: config.trainer.logger.init_args.name,
                    save_dir: config.trainer.logger.init_args.save_dir || 'logs'
                };
                loggerNode = addNode('TensorBoardLogger', 900, 200, loggerData);
            }
            
            // Parse Callbacks (Find EarlyStopping)
            if (Array.isArray(config.trainer.callbacks)) {
                const es = config.trainer.callbacks.find(c => c.class_path && c.class_path.includes('EarlyStopping'));
                if (es && es.init_args) {
                    const esData = {
                        patience: es.init_args.patience,
                        monitor: es.init_args.monitor || 'val/loss'
                    };
                    esNode = addNode('EarlyStopping', 700, 600, esData);
                }
            }
        }

        // --- Data ---
        let dataNode;
        if (config.data) {
            const init = config.data.init_args || {};
            const dataData = {
                batch_size: init.batch_size,
                num_workers: init.num_workers,
                bands: Array.isArray(init.bands) ? init.bands.join(',') : init.bands
            };
            dataNode = addNode('HLSDataModule', 500, 520, dataData);
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
        if (esNode && trainerNode) {
             state.connections.push({ sourceNode: esNode.id, sourcePort: 'callback', targetNode: trainerNode.id, targetPort: 'early_stopping' });
        }
        
        // Use multiple render strategies to ensure lines appear
        console.log("Rebuilding connections...", state.connections.length);
        renderConnections();
        
        // Failsafe for slower DOM updates
        setTimeout(() => {
            renderConnections();
        }, 100);

    } catch (err) {
        console.error(err);
        alert("⚠️ Error parsing structure: " + err.message);
    }
}
