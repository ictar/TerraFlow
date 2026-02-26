/* Import Logic from YAML */

function importYaml() {
  document.getElementById("import-file").click();
}

function handleFileImport(input) {
  const file = input.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (e) {
    try {
      const content = e.target.result;
      const config = jsyaml.load(content);

      // Use filename (without extension) as experiment name
      let filename = file.name;
      if (filename.includes(".")) {
        filename = filename.substring(0, filename.lastIndexOf("."));
      }

      reconstructGraph(config, false, filename);
      input.value = ""; // Reset
    } catch (e) {
      console.error(e);
      alert("❌ Import Failed: Invalid YAML file.\n" + e.message);
    }
  };
  reader.readAsText(file);
}

function reconstructGraph(config, silent = false, filename = null) {
  if (!silent && !confirm("Importing will clear the current canvas. Continue?"))
    return;

  // 1. Clear Canvas
  state.nodes = [];
  state.connections = [];
  nodesContainer.innerHTML = "";
  nodesContainer.appendChild(svgLayer); // CRITICAL FIX: Restore SVG layer that was wiped out
  svgLayer.innerHTML = "";

  try {
    // 1.5 Restore Global Config
    if (filename) {
      state.globalConfig.experiment_name = filename;
    } else if (config.experiment_name) {
      state.globalConfig.experiment_name = config.experiment_name;
    }
    if (config.seed_everything !== undefined) {
      state.globalConfig.seed_everything = config.seed_everything;
    }
    updateGlobalUI();

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
      trainerData.check_val_every_n_epoch =
        config.trainer.check_val_every_n_epoch;
      trainerData.log_every_n_steps = config.trainer.log_every_n_steps;
      trainerData.enable_checkpointing =
        config.trainer.enable_checkpointing === false ? "False" : "True";
      trainerData.default_root_dir =
        config.trainer.default_root_dir || "checkpoints";
      state.globalConfig.seed_everything =
        config.seed_everything !== undefined ? config.seed_everything : 0;
      // Update UI Panel
      if (typeof updateGlobalUI === "function") updateGlobalUI();
    }
    const trainerNode = addNode("TrainerConfig", 900, 380, trainerData);

    // --- Logger & Callbacks ---
    let loggerNode, esNode;
    if (config.trainer) {
      // Parse Logger(s)
      const loggerNodesData = [];
      let loggerConfigs = config.trainer.logger;

      if (loggerConfigs) {
        if (!Array.isArray(loggerConfigs)) {
          loggerConfigs = [loggerConfigs];
        }

        loggerConfigs.forEach((loggerConfig) => {
          const init = loggerConfig.init_args || {};
          let type = "TensorBoard";
          let project = "terraflow_project";
          let name = init.name || "terraflow_run";

          const cls = loggerConfig.class_path || "";
          if (cls.includes("Wandb")) {
            type = "Wandb";
            project = init.project;
          } else if (cls.includes("CSV")) {
            type = "CSV";
          } else if (cls.includes("MLFlow")) {
            type = "MLFlow";
            project = init.experiment_name;
            name = init.run_name;
          }

          const loggerData = {
            type: type,
            project: project || "terraflow_project",
            name: name,
            save_dir: init.save_dir || "logs",
          };
          loggerNodesData.push({ type: "Logger", data: loggerData });
        });
      }

      var createdLoggerNodes = [];
      loggerNodesData.forEach((item, i) => {
        const node = addNode(item.type, 900, 200 + i * 150, item.data);
        createdLoggerNodes.push(node);
      });

      // Parse Callbacks
      const callbackNodesData = [];
      if (Array.isArray(config.trainer.callbacks)) {
        config.trainer.callbacks.forEach((cb, idx) => {
          const cp = cb.class_path || "";
          if (cp.includes("EarlyStopping")) {
            const d = {
              patience: cb.init_args?.patience,
              monitor: cb.init_args?.monitor || "val/loss",
              mode: cb.init_args?.mode || "min",
            };
            callbackNodesData.push({ type: "EarlyStopping", data: d });
          } else if (cp.includes("ModelCheckpoint")) {
            const d = {
              monitor: cb.init_args?.monitor || "val/loss",
              mode: cb.init_args?.mode || "min",
              save_top_k: cb.init_args?.save_top_k || 1,
              filename: cb.init_args?.filename,
              dirpath: cb.init_args?.dirpath,
            };
            callbackNodesData.push({ type: "ModelCheckpoint", data: d });
          } else if (cp.includes("LearningRateMonitor")) {
            const d = {
              logging_interval: cb.init_args?.logging_interval || "epoch",
            };
            callbackNodesData.push({ type: "LearningRateMonitor", data: d });
          } else if (cp.includes("RichProgressBar")) {
            callbackNodesData.push({ type: "RichProgressBar", data: {} });
          } else {
            const customParamsArray = [];
            for (const [k, v] of Object.entries(cb.init_args || {})) {
              let valStr = "";
              if (v !== undefined && v !== null) {
                if (typeof v === "object") {
                  valStr = JSON.stringify(v);
                } else {
                  valStr = v.toString();
                }
              }
              customParamsArray.push({ key: k, value: valStr });
            }
            const d = {
              class_path: cp,
              customParams: customParamsArray,
            };
            callbackNodesData.push({ type: "CustomCallback", data: d });
          }
        });
      }

      // Create Callback Nodes
      // We need to store them to connect later
      var createdCallbackNodes = [];
      callbackNodesData.forEach((item, i) => {
        // Increment Y directly based on sequence index to guarantee correct Y sorting later
        const node = addNode(item.type, 700 + i * 50, 600 + i * 150, item.data);
        createdCallbackNodes.push(node);
      });
    }

    // --- Data ---
    let dataNode;
    if (config.data) {
      const init = config.data.init_args || {};
      const classPath =
        config.data.class_path || "GenericNonGeoSegmentationDataModule";

      const genericDataModules = [
        "GenericNonGeoPixelwiseRegressionDataModule",
        "GenericNonGeoSegmentationDataModule",
        "Sen1Floods11NonGeoDataModule",
        "Landsat7DataModule",
        "TorchGeoDataModule",
      ];

      if (genericDataModules.some((g) => classPath.includes(g))) {
        const dataData = {
          class_path: classPath,
          batch_size: init.batch_size,
          num_workers: init.num_workers,
          train_data_root: init.train_data_root,
          val_data_root: init.val_data_root,
          test_data_root: init.test_data_root,
          means: Array.isArray(init.means)
            ? JSON.stringify(init.means)
            : init.means,
          stds: Array.isArray(init.stds)
            ? JSON.stringify(init.stds)
            : init.stds,
          num_classes: init.num_classes,
          img_grep: init.img_grep,
          label_grep: init.label_grep,
          bands: Array.isArray(init.bands) ? init.bands.join(",") : init.bands,
          constant_scale: init.constant_scale,
          no_data_replace: init.no_data_replace,
          no_label_replace: init.no_label_replace,
          drop_last: init.drop_last === true ? "True" : "False",
          reduce_zero_label: init.reduce_zero_label === true ? "True" : "False",
          rgb_indices: Array.isArray(init.rgb_indices)
            ? init.rgb_indices.join(",")
            : init.rgb_indices,
        };
        dataNode = addNode("DataModule", 500, 520, dataData);
      } else {
        // Custom Data Module
        const customParamsArray = [];
        for (const [k, v] of Object.entries(init)) {
          if (
            !["train_transform", "val_transform", "test_transform"].includes(k)
          ) {
            let valStr = "";
            if (v !== undefined && v !== null) {
              if (typeof v === "object") {
                valStr = JSON.stringify(v);
              } else {
                valStr = v.toString();
              }
            }
            customParamsArray.push({
              key: k,
              value: valStr,
            });
          }
        }
        const dataData = {
          class_path: classPath,
          customParams: customParamsArray,
        };
        dataNode = addNode("CustomDataModule", 500, 520, dataData);
      }

      // Parse Transforms
      ["train_transform", "val_transform", "test_transform"].forEach(
        (key, keyIdx) => {
          if (Array.isArray(init[key])) {
            init[key].forEach((t, i) => {
              let type = null;
              let tData = {};
              const tInit = t.init_args || {};

              if (t.class_path.includes("Resize")) {
                type = "AlbumentationsResize";
                tData = {
                  height: tInit.height,
                  width: tInit.width,
                };
              } else if (t.class_path.includes("HorizontalFlip")) {
                type = "AlbumentationsHorizontalFlip";
                tData = { p: tInit.p };
              } else if (t.class_path.includes("VerticalFlip")) {
                type = "AlbumentationsVerticalFlip";
                tData = { p: tInit.p };
              } else if (t.class_path.includes("D4")) {
                type = "AlbumentationsD4";
                tData = { p: tInit.p };
              } else if (t.class_path.includes("ToTensorV2")) {
                type = "ToTensorV2";
              } else {
                type = "CustomTransform";
                const customParamsArray = [];
                for (const [k, v] of Object.entries(tInit)) {
                  let valStr = "";
                  if (v !== undefined && v !== null) {
                    if (typeof v === "object") {
                      valStr = JSON.stringify(v);
                    } else {
                      valStr = v.toString();
                    }
                  }
                  customParamsArray.push({ key: k, value: valStr });
                }
                tData = {
                  class_path: t.class_path,
                  customParams: customParamsArray,
                };
              }

              if (type) {
                const tNode = addNode(
                  type,
                  150 + i * 180,
                  600 + keyIdx * 250 + i * 50,
                  tData,
                );
                // Connect
                state.connections.push({
                  sourceNode: tNode.id,
                  sourcePort: "transform",
                  targetNode: dataNode.id,
                  targetPort: key,
                });
              }
            });
          }
        },
      );
    }

    // --- Model / Task ---
    let taskNode;
    let modelFactoryNode;
    let tiledNode;

    if (config.model) {
      const init = config.model.init_args || {};
      const classPath = config.model.class_path || "";

      // Task Type Detection
      let taskType = "SegmentationTask";
      if (classPath.includes("PixelwiseRegressionTask"))
        taskType = "PixelwiseRegressionTask";

      const taskData = {
        loss: init.loss,
        ignore_index: init.ignore_index,
        num_classes: (init.model_args || {}).num_classes,
        model_factory: init.model_factory, // might be string
        freeze_backbone: init.freeze_backbone === true ? "True" : "False",
        freeze_decoder: init.freeze_decoder === true ? "True" : "False",
        freeze_head: init.freeze_head === true ? "True" : "False",
        plot_on_val:
          init.plot_on_val !== undefined ? init.plot_on_val.toString() : "10",
        class_names: init.class_names
          ? Array.isArray(init.class_names)
            ? init.class_names.join(",")
            : init.class_names.toString()
          : "",
        test_dataloaders_names: init.test_dataloaders_names
          ? Array.isArray(init.test_dataloaders_names)
            ? init.test_dataloaders_names.join(",")
            : init.test_dataloaders_names.toString()
          : "",
        output_on_inference: init.output_on_inference
          ? Array.isArray(init.output_on_inference)
            ? `[${init.output_on_inference.map((s) => `'${s}'`).join(",")}]`
            : init.output_on_inference.toString()
          : "prediction",
        output_most_probable:
          init.output_most_probable === false ? "False" : "True",
        path_to_record_metrics: init.path_to_record_metrics || "",
        tiled_inference_on_testing:
          init.tiled_inference_on_testing === true ? "True" : "False",
        tiled_inference_on_validation:
          init.tiled_inference_on_validation === true ? "True" : "False",
      };
      taskNode = addNode(taskType, 500, 220, taskData);

      // Model Args (Factory) - Modular Reconstruction
      if (init.model_args) {
        const ma = init.model_args;

        // 1. Create ModelFactory (The HUB)
        // It now takes inputs, so we don't pass much data to it directly
        modelFactoryNode = addNode("ModelFactory", 100, 200, {});

        // 2. Create Backbone Node
        let modStr = "";
        if (ma.backbone_modalities !== undefined) {
          modStr =
            typeof ma.backbone_modalities === "object"
              ? JSON.stringify(ma.backbone_modalities)
              : ma.backbone_modalities.toString();
        }

        const backboneData = {
          model_name: ma.backbone,
          pretrained: ma.backbone_pretrained === false ? "False" : "True",
          backbone_modalities: modStr,
          in_channels: ma.in_chans !== undefined ? ma.in_chans : ma.in_channels,
          num_frames: ma.num_frames,
          drop_path_rate: ma.backbone_drop_path_rate,
          window_size: ma.backbone_window_size,
        };
        const backboneNode = addNode("ModelBackbone", -250, 100, backboneData);
        state.connections.push({
          sourceNode: backboneNode.id,
          sourcePort: "backbone_config",
          targetNode: modelFactoryNode.id,
          targetPort: "backbone",
        });

        // 3. Create Decoder Node
        if (ma.decoder) {
          const decoderData = {
            decoder_name: ma.decoder,
            decoder_channels: ma.decoder_channels,
            scale_modules:
              ma.decoder_scale_modules === false ? "False" : "True",
          };
          const decoderNode = addNode("ModelDecoder", -250, 250, decoderData);
          state.connections.push({
            sourceNode: decoderNode.id,
            sourcePort: "decoder_config",
            targetNode: modelFactoryNode.id,
            targetPort: "decoder",
          });
        }

        // 4. Create Head Node
        if (
          ma.head_dropout !== undefined ||
          ma.head_learned_upscale_layers !== undefined
        ) {
          const headData = {
            dropout: ma.head_dropout,
            learned_upscale_layers: ma.head_learned_upscale_layers,
            final_act: ma.head_final_act
              ? ma.head_final_act.replace("torch.nn.", "")
              : "None",
          };
          const headNode = addNode("ModelHead", -250, 400, headData);
          state.connections.push({
            sourceNode: headNode.id,
            sourcePort: "head_config",
            targetNode: modelFactoryNode.id,
            targetPort: "head",
          });
        }

        // 5. Create Neck Node (Check if explicit necks are defined)
        if (Array.isArray(ma.necks) && ma.necks.length > 0) {
          // Check if it's the default auto-gen or a custom one
          // Simple heuristic: If it has 'SelectIndices', we try to visualize it
          const selectIndicesNeck = ma.necks.find(
            (n) => n.name === "SelectIndices",
          );
          const reshapeNeck = ma.necks.find(
            (n) => n.name === "ReshapeTokensToImage",
          );

          if (selectIndicesNeck) {
            const neckData = {
              indices: (selectIndicesNeck.indices || []).join(","),
              reshape: reshapeNeck ? "True" : "False",
            };
            const neckNode = addNode("ModelNeck", -250, 550, neckData);
            state.connections.push({
              sourceNode: neckNode.id,
              sourcePort: "neck_config",
              targetNode: modelFactoryNode.id,
              targetPort: "neck",
            });
          }
        }
      }

      // Tiled Inference
      if (init.tiled_inference_parameters) {
        const ti = init.tiled_inference_parameters;
        const tiledData = {
          h_crop: ti.h_crop,
          w_crop: ti.w_crop,
          h_stride: ti.h_stride,
          w_stride: ti.w_stride,
          average_patches: ti.average_patches ? "True" : "False",
        };
        tiledNode = addNode("TiledInference", 100, 600, tiledData);
      }
    }

    // --- Optimizer ---
    let optimNode;
    if (config.optimizer) {
      const optInit = config.optimizer.init_args || {};
      const optimData = {
        type: config.optimizer.class_path.includes("SGD") ? "SGD" : "AdamW",
        lr: optInit.lr,
        weight_decay: optInit.weight_decay,
      };
      optimNode = addNode("OptimizerConfig", 100, 900, optimData);
    }

    // LR Scheduler
    let schedulerNode = null;
    if (config.lr_scheduler) {
      const schedInit = config.lr_scheduler.init_args || {};
      const schedType = config.lr_scheduler.class_path.includes("Plateau")
        ? "ReduceLROnPlateau"
        : "CosineAnnealingLR";
      const schedData = {
        type: schedType,
        monitor: schedInit.monitor || "val/loss",
        patience: schedInit.patience || 10,
        t_max: schedInit.T_max || 50,
        mode: schedInit.mode || "min",
        factor: schedInit.factor || 0.1,
        threshold: schedInit.threshold || 1e-4,
        threshold_mode: schedInit.threshold_mode || "rel",
        cooldown: schedInit.cooldown || 0,
        min_lr: schedInit.min_lr || 0,
        eps: schedInit.eps || 1e-8,
      };
      schedulerNode = addNode("LRScheduler", 100, 1050, schedData);
    }

    // 3. Reconnect
    // Force a layout update if needed (browsers usually handle this on getBoundingClientRect)

    if (modelFactoryNode && taskNode) {
      state.connections.push({
        sourceNode: modelFactoryNode.id,
        sourcePort: "model_args",
        targetNode: taskNode.id,
        targetPort: "model_args",
      });
    }
    if (optimNode && taskNode) {
      state.connections.push({
        sourceNode: optimNode.id,
        sourcePort: "optim_args",
        targetNode: taskNode.id,
        targetPort: "optim_args",
      });
    }
    if (schedulerNode && taskNode) {
      state.connections.push({
        sourceNode: schedulerNode.id,
        sourcePort: "scheduler_args",
        targetNode: taskNode.id,
        targetPort: "scheduler_args",
      });
    }
    if (taskNode && trainerNode) {
      state.connections.push({
        sourceNode: taskNode.id,
        sourcePort: "task",
        targetNode: trainerNode.id,
        targetPort: "task",
      });
    }
    if (dataNode && trainerNode) {
      state.connections.push({
        sourceNode: dataNode.id,
        sourcePort: "datamodule",
        targetNode: trainerNode.id,
        targetPort: "datamodule",
      });
    }
    if (createdLoggerNodes && createdLoggerNodes.length > 0 && trainerNode) {
      createdLoggerNodes.forEach((node) => {
        state.connections.push({
          sourceNode: node.id,
          sourcePort: "logger",
          targetNode: trainerNode.id,
          targetPort: "logger",
        });
      });
    }
    if (tiledNode && taskNode) {
      state.connections.push({
        sourceNode: tiledNode.id,
        sourcePort: "tiled_inference",
        targetNode: taskNode.id,
        targetPort: "tiled_inference",
      });
    }
    if (createdCallbackNodes.length > 0 && trainerNode) {
      createdCallbackNodes.forEach((cbNode) => {
        state.connections.push({
          sourceNode: cbNode.id,
          sourcePort: "callback",
          targetNode: trainerNode.id,
          targetPort: "callbacks",
        });
      });
    }

    // Use multiple render strategies to ensure lines appear
    console.log("Rebuilding connections...", state.connections.length);
    renderConnections();

    // Failsafe for slower DOM updates
    setTimeout(() => {
      renderConnections();
      if (typeof autoLayout === "function") autoLayout();
    }, 100);
  } catch (err) {
    console.error(err);
    alert("⚠️ Error parsing structure: " + err.message);
  }
}
