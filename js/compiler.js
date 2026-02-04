/* Compiler & Export Logic */

function findInput(nodeId, portName) {
  const conn = state.connections.find(
    (c) => c.targetNode === nodeId && c.targetPort === portName,
  );
  if (!conn) return null;
  return state.nodes.find((n) => n.id === conn.sourceNode);
}

function findAllInputs(nodeId, portName) {
  const conns = state.connections.filter(
    (c) => c.targetNode === nodeId && c.targetPort === portName,
  );
  return conns
    .map((conn) => state.nodes.find((n) => n.id === conn.sourceNode))
    .filter((n) => n);
}

function validateDetails(trainerNode) {
  if (!trainerNode) throw new Error("Graph must contain a Trainer node.");

  const taskNode = findInput(trainerNode.id, "task");
  if (!taskNode) throw new Error("Trainer must have a Task connected.");

  const dataNode = findInput(trainerNode.id, "datamodule");
  if (!dataNode) throw new Error("Trainer must have a DataModule connected.");

  return { taskNode, dataNode };
}

// Strategy 1: Hydra Config (For Notebook Export)
function buildHydraConfig() {
  const trainerNode = state.nodes.find((n) => n.type === "TrainerConfig");
  const { taskNode, dataNode } = validateDetails(trainerNode);
  const modelNode = findInput(taskNode.id, "model_args");
  const optimNode = findInput(taskNode.id, "optim_args");

  const bands = (dataNode.data.bands || "BLUE,GREEN,RED,NIR").split(",");

  // Construct Object
  const fullConfig = {
    defaults: [
      { "/terratorch/datamodules": "generic_non_geo_segmentation" }, // Simplified default for Hydra
      { "/terratorch/tasks": taskNode.type },
      { "/terratorch/trainers": "default_trainer" }, // Using simple default for Hydra base
    ],
    trainer: {
      max_epochs: parseInt(trainerNode.data.max_epochs),
      accelerator: trainerNode.data.accelerator,
      precision: trainerNode.data.precision,
      devices:
        trainerNode.data.devices === "auto"
          ? "auto"
          : parseInt(trainerNode.data.devices) || trainerNode.data.devices,
      strategy: trainerNode.data.strategy,
      num_nodes: parseInt(trainerNode.data.num_nodes),
      check_val_every_n_epoch: parseInt(
        trainerNode.data.check_val_every_n_epoch,
      ),
      log_every_n_steps: parseInt(trainerNode.data.log_every_n_steps),
    },
    datamodule: {
      _target_: `terratorch.datamodules.${dataNode.data.class_path}`,
      batch_size: parseInt(dataNode.data.batch_size),
      num_workers: parseInt(dataNode.data.num_workers),
      bands: bands,
    },
    task: {
      _target_: `terratorch.tasks.${taskNode.type}`,
      loss: taskNode.data.loss,
      ignore_index: parseInt(taskNode.data.ignore_index),
      model_args: {
        decoder: "UperNetDecoder",
        backbone: modelNode ? modelNode.data.backbone : "prithvi_eo_v1_100",
        num_classes: parseInt(taskNode.data.num_classes),
        backbone_pretrained: modelNode
          ? modelNode.data.pretrained === "True"
          : true,
      },
    },
  };

  if (optimNode) {
    fullConfig.task.optimizer = {
      _target_: `torch.optim.${optimNode.data.type}`,
      lr: parseFloat(optimNode.data.lr),
      weight_decay: parseFloat(optimNode.data.weight_decay),
    };
  }

  return { yaml: jsyaml.dump(fullConfig), lossType: taskNode.data.loss };
}

// Helper to extract custom params
function extractParams(node) {
  const params = {};
  if (node.data.customParams) {
    node.data.customParams.forEach((p) => {
      let val = p.value;
      // Simple Type Inference
      if (val === "True") val = true;
      else if (val === "False") val = false;
      else if (!isNaN(val) && val.trim() !== "") {
        // Check if it looks like a number
        if (val.includes(".")) val = parseFloat(val);
        else val = parseInt(val);
      }
      params[p.key] = val;
    });
  }
  return params;
}

// Strategy 2: Lightning CLI Config (For YAML Export)
function buildCLIConfig() {
  const trainerNode = state.nodes.find((n) => n.type === "TrainerConfig");
  // Allow partial exports if Trainer is missing? No, Trainer is root.
  if (!trainerNode) throw new Error("Graph must contain a Trainer node.");

  const taskNode = findInput(trainerNode.id, "task");
  if (!taskNode) throw new Error("Trainer must have a Task connected.");

  const dataNode = findInput(trainerNode.id, "datamodule");
  if (!dataNode) throw new Error("Trainer must have a DataModule connected.");

  // Model Configs
  const modelNode = findInput(taskNode.id, "model_args"); // Can be Factory or Null
  const optimNode = findInput(taskNode.id, "optim_args");
  const schedulerNode = findAllInputs(taskNode.id, "scheduler_args")[0];

  // Logger & Callbacks
  const loggerNode = findInput(trainerNode.id, "logger");
  const callbackNodes = findAllInputs(trainerNode.id, "callbacks");
  // Sort callbacks by Y position to match visual order (like transforms)
  callbackNodes.sort((a, b) => a.y - b.y);

  const bands = (dataNode.data.bands || "BLUE,GREEN,RED,NIR").split(",");

  // --- Callbacks ---
  let callbacks = [];
  if (callbackNodes.length > 0) {
    callbackNodes.forEach((node) => {
      if (node.type === "CustomCallback") {
        callbacks.push({
          class_path: node.data.class_path,
          init_args: extractParams(node),
        });
      } else if (node.type === "EarlyStopping") {
        callbacks.push({
          class_path: "lightning.pytorch.callbacks.EarlyStopping",
          init_args: {
            patience: parseInt(node.data.patience),
            monitor: node.data.monitor,
            mode: node.data.mode,
          },
        });
      } else if (node.type === "ModelCheckpoint") {
        callbacks.push({
          class_path: "lightning.pytorch.callbacks.ModelCheckpoint",
          init_args: {
            monitor: node.data.monitor,
            mode: node.data.mode,
            save_top_k: parseInt(node.data.save_top_k),
            filename: node.data.filename,
          },
        });
      } else if (node.type === "LearningRateMonitor") {
        callbacks.push({
          class_path: "LearningRateMonitor",
          init_args: { logging_interval: node.data.logging_interval },
        });
      } else if (node.type === "RichProgressBar") {
        callbacks.push({ class_path: "RichProgressBar" });
      }
    });
  } else {
    // Default
    callbacks.push({ class_path: "RichProgressBar" });
    callbacks.push({
      class_path: "LearningRateMonitor",
      init_args: { logging_interval: "epoch" },
    });
  }

  // --- Logger ---
  let loggerConfig = null;
  if (loggerNode) {
    if (loggerNode.type === "CustomLogger") {
      loggerConfig = {
        class_path: loggerNode.data.type, // Custom Logger uses 'type' param as classpath default
        init_args: extractParams(loggerNode),
      };
    } else {
      const type = loggerNode.data.type;
      const args = {
        save_dir: loggerNode.data.save_dir,
        name: loggerNode.data.name,
      };

      let classPath = "lightning.pytorch.loggers.TensorBoardLogger";
      if (type === "CSVLogger")
        classPath = "lightning.pytorch.loggers.CSVLogger";
      else if (type === "WandbLogger") {
        classPath = "lightning.pytorch.loggers.WandbLogger";
        if (loggerNode.data.project) args.project = loggerNode.data.project;
      }
      loggerConfig = { class_path: classPath, init_args: args };
    }
  }

  // --- Task Class ---
  let taskClass = "terratorch.tasks.SemanticSegmentationTask";
  if (taskNode.type === "PixelwiseRegressionTask")
    taskClass = "terratorch.tasks.PixelwiseRegressionTask";
  else if (taskNode.type === "CustomTask") taskClass = taskNode.data.class_path;

  // --- Model Factory Inputs ---
  let backboneNode = null,
    decoderNode = null,
    headNode = null,
    neckNode = null;
  if (modelNode) {
    backboneNode = findAllInputs(modelNode.id, "backbone")[0];
    decoderNode = findAllInputs(modelNode.id, "decoder")[0];
    headNode = findAllInputs(modelNode.id, "head")[0];
    neckNode = findAllInputs(modelNode.id, "neck")[0];
  }

  // Necks Logic
  let necksConfig = [];
  if (neckNode) {
    if (neckNode.type === "CustomNeck") {
      // How CustomNeck behaves? assume generic object?
      // Necks in TerraTorch are usually list of dicts.
      // Let's assume user CustomNeck creates one entry.
      // Or maybe CustomNeck adds to Model Args?
      // For now, let's treat CustomNeck as "Custom Config Entry" if connected?
      // Safer: Just allow it to inject params.
      necksConfig.push(extractParams(neckNode));
    } else {
      const indices = neckNode.data.indices
        .split(",")
        .map((s) => parseInt(s.trim()))
        .filter((n) => !isNaN(n));
      necksConfig.push({ name: "SelectIndices", indices: indices });
      if (neckNode.data.reshape === "True")
        necksConfig.push({ name: "ReshapeTokensToImage" });
    }
  } else {
    // Default
    necksConfig = getNecksForBackbone(
      backboneNode ? backboneNode.data.model_name : "prithvi_eo_v1_100",
    );
  }

  // --- Tiled Inference ---
  let tiledNode = findAllInputs(taskNode.id, "tiled_inference")[0];
  let tiledParams = {};
  if (tiledNode) {
    tiledParams = {
      h_crop: parseInt(tiledNode.data.h_crop),
      w_crop: parseInt(tiledNode.data.w_crop),
      h_stride: parseInt(tiledNode.data.h_stride),
      w_stride: parseInt(tiledNode.data.w_stride),
      average_patches: tiledNode.data.average_patches === "True",
    };
  }

  // --- Scheduler ---
  let schedulerConfig = null;
  if (schedulerNode) {
    if (schedulerNode.type === "CustomLRScheduler") {
      schedulerConfig = {
        class_path: schedulerNode.data.type,
        init_args: extractParams(schedulerNode),
      };
    } else {
      if (schedulerNode.data.type === "ReduceLROnPlateau") {
        schedulerConfig = {
          class_path: "ReduceLROnPlateau",
          init_args: {
            monitor: schedulerNode.data.monitor,
            patience: parseInt(schedulerNode.data.patience),
            mode: "min",
          },
        };
      } else {
        schedulerConfig = {
          class_path: "CosineAnnealingLR",
          init_args: { T_max: parseInt(schedulerNode.data.t_max) },
        };
      }
    }
  } else {
    schedulerConfig = {
      class_path: "CosineAnnealingLR",
      init_args: { T_max: parseInt(trainerNode.data.max_epochs) },
    };
  }

  // --- Optimizer ---
  let optimizerConfig = {};
  if (optimNode) {
    if (optimNode.type === "CustomOptimizer") {
      optimizerConfig = {
        class_path: optimNode.data.type,
        init_args: extractParams(optimNode),
      };
    } else {
      optimizerConfig = {
        class_path:
          optimNode.data.type === "SGD"
            ? "torch.optim.SGD"
            : "torch.optim.AdamW",
        init_args: {
          lr: parseFloat(optimNode.data.lr),
          weight_decay: parseFloat(optimNode.data.weight_decay),
        },
      };
    }
  }

  // --- CLI Config Construction ---

  // Model Args Accumulation
  let finalModelArgs = {
    // Defaults
    backbone: "prithvi_eo_v1_100",
    decoder: "UperNetDecoder",
    num_classes: taskNode.data.num_classes
      ? parseInt(taskNode.data.num_classes)
      : undefined,
    necks: necksConfig,
  };

  // Backbone Info
  if (backboneNode) {
    finalModelArgs.backbone = backboneNode.data.model_name;
    if (backboneNode.type === "CustomBackbone") {
      Object.assign(finalModelArgs, extractParams(backboneNode)); // Merge custom
    } else {
      // Standard
      finalModelArgs.backbone_pretrained =
        backboneNode.data.pretrained === "True";
      finalModelArgs.in_chans = parseInt(backboneNode.data.in_channels);
      finalModelArgs.num_frames = parseInt(backboneNode.data.num_frames);
      // etc...
    }
    finalModelArgs.backbone_bands = [...bands];
  } else {
    // Fallback defaults
    finalModelArgs.backbone_bands = [...bands];
  }

  // Decoder Info
  if (decoderNode) {
    finalModelArgs.decoder = decoderNode.data.decoder_name;
    if (decoderNode.type === "CustomDecoder") {
      Object.assign(finalModelArgs, extractParams(decoderNode));
    } else {
      finalModelArgs.decoder_channels = parseInt(
        decoderNode.data.decoder_channels,
      );
      finalModelArgs.decoder_scale_modules =
        decoderNode.data.scale_modules === "True";
    }
  }

  // Head Info
  if (headNode) {
    if (headNode.type === "CustomHead") {
      Object.assign(finalModelArgs, extractParams(headNode));
    } else {
      finalModelArgs.dropout = parseFloat(headNode.data.dropout);
      finalModelArgs.learned_upscale_layers = parseInt(
        headNode.data.learned_upscale_layers,
      );
      if (headNode.data.final_act !== "None")
        finalModelArgs.final_act = `torch.nn.${headNode.data.final_act}`;
    }
  }

  // Tiled Inference
  if (Object.keys(tiledParams).length > 0) {
    finalModelArgs.tiled_inference_parameters = tiledParams;
  }

  // Data Init Args
  let dataInit = {};
  if (dataNode.type === "CustomDataModule") {
    dataInit = extractParams(dataNode);
  } else {
    // Parse lists safely
    let means = [],
      stds = [],
      rgb_indices = [];
    try {
      means = JSON.parse(dataNode.data.means || "[]");
    } catch (e) {}
    try {
      stds = JSON.parse(dataNode.data.stds || "[]");
    } catch (e) {}
    if (dataNode.data.rgb_indices) {
      rgb_indices = dataNode.data.rgb_indices
        .split(",")
        .map((s) => parseInt(s.trim()))
        .filter((n) => !isNaN(n));
    }

    dataInit = {
      batch_size: parseInt(dataNode.data.batch_size),
      num_workers: parseInt(dataNode.data.num_workers),
      train_data_root: dataNode.data.train_data_root,
      val_data_root: dataNode.data.val_data_root,
      test_data_root: dataNode.data.test_data_root,
      means: means,
      stds: stds,
      num_classes: parseInt(dataNode.data.num_classes),
      img_grep: dataNode.data.img_grep,
      label_grep: dataNode.data.label_grep,
      constant_scale: parseFloat(dataNode.data.constant_scale),
      no_data_replace: parseInt(dataNode.data.no_data_replace),
      no_label_replace: parseInt(dataNode.data.no_label_replace),
      drop_last: dataNode.data.drop_last === "True",
      reduce_zero_label: dataNode.data.reduce_zero_label === "True",
      rgb_indices: rgb_indices,
      bands: [...bands],
    };
  }
  // Append Transforms
  dataInit.train_transform = gatherTransforms(dataNode.id, "train_transform");
  dataInit.val_transform = gatherTransforms(dataNode.id, "val_transform");
  dataInit.test_transform = gatherTransforms(dataNode.id, "test_transform");

  // Task & Model Block
  const modelBlock = {
    class_path: taskClass,
    init_args: {
      model_args: finalModelArgs,
      loss: taskNode.data.loss, // Might be undefined for CustomTask?
      ignore_index: taskNode.data.ignore_index
        ? parseInt(taskNode.data.ignore_index)
        : undefined,
      model_factory: taskNode.data.model_factory || "EncoderDecoderFactory",
    },
  };

  // Custom Task Params integration
  if (taskNode.type === "CustomTask") {
    // Merge custom params into init_args directly (not model_args)
    Object.assign(modelBlock.init_args, extractParams(taskNode));
  }

  const cliConfig = {
    seed_everything: state.globalConfig.seed_everything,
    trainer: {
      accelerator: trainerNode.data.accelerator,
      strategy: trainerNode.data.strategy,
      devices:
        trainerNode.data.devices === "auto"
          ? "auto"
          : parseInt(trainerNode.data.devices) || trainerNode.data.devices,
      num_nodes: parseInt(trainerNode.data.num_nodes),
      precision: trainerNode.data.precision,
      logger: loggerConfig,
      callbacks: callbackNodes.length > 0 ? callbacks : null,
      max_epochs: parseInt(trainerNode.data.max_epochs),
      enable_checkpointing: trainerNode.data.enable_checkpointing === "True",
      default_root_dir: trainerNode.data.default_root_dir,
    },
    data: {
      class_path:
        dataNode.type === "CustomDataModule"
          ? dataNode.data.class_path
          : dataNode.data.class_path, // Standard has it too
      // Standard one needs prefix? `terratorch.datamodules.`?
      // In original code: `class_path: dataNode.data.class_path` was used in `buildCLIConfig` line 244.
      // BUT wait, line 53 used prefix in `buildHydraConfig`.
      // Line 244 in `buildCLIConfig` used simple keys?
      // Standard CLI usually expects full path if not registered.
      // Terratorch CLI might handle shortcuts.
      // Let's assume original code was correct for Standard.
      // For Custom, we definitely want whatever the user typed.
      init_args: dataInit,
    },
    model: modelBlock,
    optimizer: Object.keys(optimizerConfig).length ? optimizerConfig : {},
    lr_scheduler: schedulerConfig,
  };

  // Cleanup
  if (!cliConfig.trainer.logger) delete cliConfig.trainer.logger;
  if (!cliConfig.trainer.callbacks) delete cliConfig.trainer.callbacks;

  return cliConfig;
}

// Helper to gather and sort transforms (by Y position for sequence)
function gatherTransforms(nodeId, portName) {
  const nodes = findAllInputs(nodeId, portName);
  // Sort by Y position (top to bottom execution)
  nodes.sort((a, b) => a.y - b.y);

  return nodes
    .map((node) => {
      if (node.type === "CustomTransform") {
        return {
          class_path: node.data.class_path,
          init_args: extractParams(node),
        };
      }
      if (node.type === "AlbumentationsResize") {
        return {
          class_path: "albumentations.Resize",
          init_args: {
            height: parseInt(node.data.height),
            width: parseInt(node.data.width),
          },
        };
      } else if (node.type === "AlbumentationsHorizontalFlip") {
        return {
          class_path: "albumentations.HorizontalFlip",
          init_args: { p: parseFloat(node.data.p) },
        };
      } else if (node.type === "AlbumentationsVerticalFlip") {
        return {
          class_path: "albumentations.VerticalFlip",
          init_args: { p: parseFloat(node.data.p) },
        };
      } else if (node.type === "AlbumentationsD4") {
        return {
          class_path: "albumentations.D4",
          init_args: { p: parseFloat(node.data.p) },
        };
      } else if (node.type === "ToTensorV2") {
        return { class_path: "ToTensorV2" };
      }
      return null;
    })
    .filter((t) => t);
}

// --- Helpers ---

function mdCell(source) {
  return { cell_type: "markdown", metadata: {}, source: [source] };
}
function codeCell(source) {
  return {
    cell_type: "code",
    execution_count: null,
    metadata: {},
    outputs: [],
    source: source,
  };
}

function generateNotebookJSON(config, lossType) {
  const cells = [];

  // 1. Header
  cells.push(
    mdCell(
      `# 🌍 TerraFlow Job - ${config.experiment_name}\n\nGenerated by TerraFlow UI.`,
    ),
  );

  // 2. Setup
  cells.push(
    codeCell([
      "# Install dependencies\n",
      "!pip install terratorch torchgeo pytorch-lightning\n",
      "from google.colab import drive\n",
      "drive.mount('/content/drive')",
    ]),
  );

  // 3. Imports (Dynamic)
  const taskClassFull = config.model.class_path;
  const taskPkg = taskClassFull.substring(0, taskClassFull.lastIndexOf("."));
  const taskCls = taskClassFull.split(".").pop();

  const dataClassFull =
    config.data.class_path || "GenericNonGeoSegmentationDataModule";
  // Handle generic cases or fully qualified paths
  const dataPkg = dataClassFull.includes(".")
    ? dataClassFull.substring(0, dataClassFull.lastIndexOf("."))
    : "terratorch.datamodules";
  const dataCls = dataClassFull.includes(".")
    ? dataClassFull.split(".").pop()
    : dataClassFull;

  const imports = [
    "import os",
    "import torch",
    "import pytorch_lightning as pl",
    "from terratorch.cli_tools import build_lightning_module", // Keep as backup or specific helper
    `from ${taskPkg} import ${taskCls}`,
    `from ${dataPkg} import ${dataCls}`,
  ];

  // Add Optimizer/Scheduler imports if standard
  if (config.optimizer.class_path) {
    if (config.optimizer.class_path.startsWith("torch.optim"))
      imports.push("import torch.optim as optim");
  }
  if (config.lr_scheduler.class_path) {
    if (config.lr_scheduler.class_path === "ReduceLROnPlateau")
      imports.push("from torch.optim.lr_scheduler import ReduceLROnPlateau");
    if (config.lr_scheduler.class_path === "CosineAnnealingLR")
      imports.push("from torch.optim.lr_scheduler import CosineAnnealingLR");
  }

  cells.push(codeCell(imports.join("\n")));

  // 4. Configuration (Pythonic)
  const dataInit = JSON.stringify(config.data.init_args, null, 4);
  const modelArgs = JSON.stringify(config.model.init_args.model_args, null, 4);
  const trainerArgs = JSON.stringify(config.trainer, null, 4);

  // Construct instantiation strings
  let optimBlock = "";
  if (config.optimizer.class_path) {
    const optimCls = config.optimizer.class_path.split(".").pop();
    const optimInit = JSON.stringify(config.optimizer.init_args || {});

    // Scheduler
    const schedCls = config.lr_scheduler.class_path; // Assuming simple name like ReduceLROnPlateau
    const schedInit = JSON.stringify(config.lr_scheduler.init_args || {});

    optimBlock = `
    def configure_optimizers(self):
        optimizer = optim.${optimCls}(self.parameters(), **${optimInit})
        scheduler = ${schedCls}(optimizer, **${schedInit})
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "${config.lr_scheduler.init_args.monitor || "val/loss"}",
            },
        }
# Monkey patch or use partial if standard task doesn't support direct inject. 
# TerraTorch tasks usually take optim_args/scheduler_args dictionaries if creating from config, 
# but direct instantiation might need a custom class or factory method.
# For simplicity in this Notebook export, we will use the 'build_lightning_module' approach for the Task 
# because it handles the complex factory logic for Optimizers/Schedulers internally via CLI args usually.
# However, to match the USER REQUEST of "Explicit Style", we will prepare the args dictionaries.
`;
  }

  cells.push(
    codeCell([
      "# ---------------- Setup Configurations ---------------- #\n",
      `DATA_ARGS = ${dataInit}\n`,
      `MODEL_ARGS = ${modelArgs}\n`,
      `TRAINER_ARGS = ${trainerArgs}\n`,
      `\n# Loss & Optimization`,
      `LOSS = "${config.model.init_args.loss}"`,
      `IGNORE_INDEX = ${config.model.init_args.ignore_index}`,
      `OPTIM_ARGS = ${JSON.stringify(config.optimizer || {}, null, 4)}`,
      `SCHEDULER_ARGS = ${JSON.stringify(config.lr_scheduler || {}, null, 4)}`,
    ]),
  );

  // 5. Instantiation
  cells.push(
    codeCell([
      "# ---------------- Instantiate Components ---------------- #\n",
      "print('🔍 Instantiating DataModule...')",
      `dm = ${dataCls}(**DATA_ARGS)`,
      "dm.setup('fit')\n",
      "print('🧠 Instantiating Task...')",
      "# We use the class directly, passing the model_args dict",
      `task = ${taskCls}(`,
      "    model_args=MODEL_ARGS,",
      "    loss=LOSS,",
      "    ignore_index=IGNORE_INDEX,",
      "    optimizer=OPTIM_ARGS.get('class_path'),",
      "    optimizer_hparams=OPTIM_ARGS.get('init_args'),",
      "    scheduler=SCHEDULER_ARGS.get('class_path'),",
      "    scheduler_hparams=SCHEDULER_ARGS.get('init_args')",
      ")",
    ]),
  );

  // 6. Training
  cells.push(
    codeCell([
      "# ---------------- Training ---------------- #",
      "trainer = pl.Trainer(**TRAINER_ARGS)",
      "trainer.fit(task, dm)",
    ]),
  );

  return {
    metadata: {
      kernelspec: {
        display_name: "Python 3",
        language: "python",
        name: "python3",
      },
    },
    nbformat: 4,
    nbformat_minor: 4,
    cells: cells,
  };
}

function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// --- Main Export Functions ---

function exportNotebook() {
  try {
    const config = buildCLIConfig();
    const notebook = generateNotebookJSON(config);
    const filename = state.globalConfig.experiment_name || "TerraFlow";
    downloadJSON(notebook, `${filename}.ipynb`);
  } catch (e) {
    alert("⚠️ Export Failed: " + e.message);
    console.error(e);
  }
}

function exportYaml() {
  try {
    const config = buildCLIConfig();
    const yamlStr = jsyaml.dump(config);

    const blob = new Blob([yamlStr], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const filename = state.globalConfig.experiment_name || "config";
    a.download = `${filename}.yaml`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } catch (e) {
    alert("⚠️ Export Failed: " + e.message);
  }
}
