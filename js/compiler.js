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

  return {
    yaml: jsyaml.dump(fullConfig, { lineWidth: -1 }),
    lossType: taskNode.data.loss,
  };
}

function extractParams(node) {
  const params = {};
  if (node.data.customParams) {
    node.data.customParams.forEach((p) => {
      let val = p.value;
      // Simple Type Inference
      if (typeof val === "string") {
        const trimmed = val.trim();
        const lowered = trimmed.toLowerCase();
        if (lowered === "true") val = true;
        else if (lowered === "false") val = false;
        else if (!isNaN(val) && trimmed !== "") {
          // Check if it looks like a number
          if (trimmed.includes(".")) val = parseFloat(trimmed);
          else val = parseInt(trimmed);
        } else if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
          // Attempt to parse JSON strings back into objects/arrays
          try {
            val = JSON.parse(trimmed);
          } catch (e) {
            // Keep as string if it's not valid JSON
          }
        }
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
  const loggerNodes = findAllInputs(trainerNode.id, "logger");
  // Sort loggers to maintain order
  loggerNodes.sort((a, b) => a.y - b.y);
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
            dirpath: node.data.dirpath,
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
  if (loggerNodes.length > 0) {
    const configs = [];
    loggerNodes.forEach((loggerNode) => {
      if (loggerNode.type === "CustomLogger") {
        configs.push({
          class_path: loggerNode.data.type, // Custom Logger uses 'type' param as classpath default
          init_args: extractParams(loggerNode),
        });
      } else {
        const type = loggerNode.data.type;
        const args = {
          save_dir: loggerNode.data.save_dir,
          name: loggerNode.data.name,
        };

        let classPath = "lightning.pytorch.loggers.TensorBoardLogger";
        if (type === "CSVLogger" || type === "CSV" || type === "CSV Logger")
          classPath = "lightning.pytorch.loggers.CSVLogger";
        else if (type === "WandbLogger" || type === "Wandb") {
          classPath = "lightning.pytorch.loggers.WandbLogger";
          if (loggerNode.data.project) args.project = loggerNode.data.project;
        } else if (type === "MLFlow") {
          classPath = "lightning.pytorch.loggers.MLFlowLogger";
        }
        configs.push({ class_path: classPath, init_args: args });
      }
    });

    // PyTorch Lightning supports a list of loggers
    if (configs.length === 1) {
      loggerConfig = configs[0];
    } else {
      loggerConfig = configs;
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
          class_path: "torch.optim.lr_scheduler.ReduceLROnPlateau",
          init_args: {
            monitor: schedulerNode.data.monitor,
            mode: schedulerNode.data.mode,
            factor: parseFloat(schedulerNode.data.factor),
            patience: parseInt(schedulerNode.data.patience),
            threshold: parseFloat(schedulerNode.data.threshold),
            threshold_mode: schedulerNode.data.threshold_mode,
            cooldown: parseInt(schedulerNode.data.cooldown),
            min_lr: parseFloat(schedulerNode.data.min_lr),
            eps: parseFloat(schedulerNode.data.eps),
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

  // Model Args Accumulation (Structured to keep related args together)
  let finalModelArgs = {};

  // Backbone Info
  if (backboneNode) {
    if (backboneNode.type === "CustomBackbone") {
      finalModelArgs.backbone =
        backboneNode.data.model_name || "prithvi_eo_v1_100";
      Object.assign(finalModelArgs, extractParams(backboneNode)); // Merge custom
      finalModelArgs.backbone_bands = [...bands];
    } else {
      // Standard
      finalModelArgs.backbone = backboneNode.data.model_name;
      finalModelArgs.backbone_pretrained =
        backboneNode.data.pretrained === "True";
      finalModelArgs.backbone_bands = [...bands];

      if (backboneNode.data.backbone_modalities) {
        let modStr = backboneNode.data.backbone_modalities;
        if (typeof modStr !== "string") modStr = JSON.stringify(modStr);
        modStr = modStr.trim();
        if (modStr !== "") {
          try {
            finalModelArgs.backbone_modalities = JSON.parse(modStr);
          } catch (e) {
            finalModelArgs.backbone_modalities = modStr;
          }
        }
      }

      finalModelArgs.in_chans = parseInt(backboneNode.data.in_channels);
      finalModelArgs.num_frames = parseInt(backboneNode.data.num_frames);
      // etc...
    }
  } else {
    // Fallback defaults
    finalModelArgs.backbone = "prithvi_eo_v1_100";
    finalModelArgs.backbone_bands = [...bands];
  }

  if (necksConfig && necksConfig.length > 0) {
    finalModelArgs.necks = necksConfig;
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
  } else {
    finalModelArgs.decoder = "UperNetDecoder";
  }

  if (taskNode.data.num_classes) {
    finalModelArgs.num_classes = parseInt(taskNode.data.num_classes);
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

  if (dataInit.train_transform.length === 0) delete dataInit.train_transform;
  if (dataInit.val_transform.length === 0) delete dataInit.val_transform;
  if (dataInit.test_transform.length === 0) delete dataInit.test_transform;

  // Task Init Block - Ordered to match typical python __init__ signature
  let taskInitArgs = {};

  // 1. model_factory
  if (
    taskNode.data.model_factory &&
    taskNode.data.model_factory !== "EncoderDecoderFactory"
  ) {
    taskInitArgs.model_factory = taskNode.data.model_factory;
  }

  // 2. model_args
  taskInitArgs.model_args = finalModelArgs;

  // 3. loss
  if (taskNode.data.loss && taskNode.data.loss !== "ce") {
    taskInitArgs.loss = taskNode.data.loss;
  }

  // 4. ignore_index
  if (
    taskNode.data.ignore_index !== undefined &&
    taskNode.data.ignore_index !== "-1" &&
    taskNode.data.ignore_index !== -1
  ) {
    taskInitArgs.ignore_index = parseInt(taskNode.data.ignore_index);
  }

  // 5. freeze arguments
  if (taskNode.data.freeze_backbone === "True")
    taskInitArgs.freeze_backbone = true;
  if (taskNode.data.freeze_decoder === "True")
    taskInitArgs.freeze_decoder = true;
  if (taskNode.data.freeze_head === "True") taskInitArgs.freeze_head = true;

  // 6. other configs
  if (
    taskNode.data.plot_on_val &&
    taskNode.data.plot_on_val.toString() !== "10"
  ) {
    let pv = taskNode.data.plot_on_val.toString().trim();
    if (pv.toLowerCase() === "true") taskInitArgs.plot_on_val = true;
    else if (pv.toLowerCase() === "false") taskInitArgs.plot_on_val = false;
    else if (!isNaN(pv) && pv !== "") taskInitArgs.plot_on_val = parseInt(pv);
    else taskInitArgs.plot_on_val = pv;
  }

  if (taskNode.data.class_names && taskNode.data.class_names.trim() !== "") {
    let cn = taskNode.data.class_names.trim();
    if (cn.startsWith("[") && cn.endsWith("]")) {
      try {
        taskInitArgs.class_names = JSON.parse(cn.replace(/'/g, '"'));
      } catch (e) {
        taskInitArgs.class_names = cn.split(",").map((s) => s.trim());
      }
    } else {
      taskInitArgs.class_names = cn.split(",").map((s) => s.trim());
    }
  }

  if (
    taskNode.data.test_dataloaders_names &&
    taskNode.data.test_dataloaders_names.trim() !== ""
  ) {
    let tdn = taskNode.data.test_dataloaders_names.trim();
    if (tdn.startsWith("[") && tdn.endsWith("]")) {
      try {
        taskInitArgs.test_dataloaders_names = JSON.parse(
          tdn.replace(/'/g, '"'),
        );
      } catch (e) {
        taskInitArgs.test_dataloaders_names = tdn
          .split(",")
          .map((s) => s.trim());
      }
    } else {
      taskInitArgs.test_dataloaders_names = tdn.split(",").map((s) => s.trim());
    }
  }

  if (
    taskNode.data.output_on_inference &&
    taskNode.data.output_on_inference.trim() !== "prediction"
  ) {
    let ooi = taskNode.data.output_on_inference.trim();
    if (ooi.startsWith("[") && ooi.endsWith("]")) {
      try {
        taskInitArgs.output_on_inference = JSON.parse(ooi.replace(/'/g, '"'));
      } catch (e) {
        taskInitArgs.output_on_inference = ooi.split(",").map((s) => s.trim());
      }
    } else {
      taskInitArgs.output_on_inference = ooi;
    }
  }

  if (taskNode.data.output_most_probable === "False")
    taskInitArgs.output_most_probable = false;

  if (
    taskNode.data.path_to_record_metrics &&
    taskNode.data.path_to_record_metrics.trim() !== ""
  ) {
    taskInitArgs.path_to_record_metrics =
      taskNode.data.path_to_record_metrics.trim();
  }

  if (taskNode.data.tiled_inference_on_testing === "True")
    taskInitArgs.tiled_inference_on_testing = true;
  if (taskNode.data.tiled_inference_on_validation === "True")
    taskInitArgs.tiled_inference_on_validation = true;

  // Custom Task Params integration
  if (taskNode.type === "CustomTask") {
    Object.assign(taskInitArgs, extractParams(taskNode));
  }

  const modelBlock = {
    class_path: taskClass,
    init_args: taskInitArgs,
  };

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
      check_val_every_n_epoch: parseInt(
        trainerNode.data.check_val_every_n_epoch,
      ),
      log_every_n_steps: parseInt(trainerNode.data.log_every_n_steps),
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
    let yamlStr = jsyaml.dump(config, { lineWidth: -1 });

    // JS object keys are strictly strings. jsyaml will dump a key like "5" as '5':
    // If we want it to be a real integer key in the python side (e.g. 5: 0),
    // we use a regex to strip surrounding quotes if the key is purely digits.
    yamlStr = yamlStr.replace(/^(\s*)['"](\d+)['"]:/gm, "$1$2:");

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
