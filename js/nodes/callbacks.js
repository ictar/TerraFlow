window.NODE_TYPES = window.NODE_TYPES || {};
Object.assign(window.NODE_TYPES, {
  // [CALLBACK] - Rose Theme
  // [CALLBACK] - Rose Theme
  EarlyStopping: {
    title: "Early Stopping",
    color: "#f43f5e", // Rose 500
    inputs: [],
    outputs: ["callback"],
    params: [
      { key: "patience", label: "PATIENCE", type: "number", default: 20 },
      { key: "monitor", label: "MONITOR", type: "text", default: "val/loss" },
      {
        key: "mode",
        label: "MODE",
        type: "select",
        options: ["min", "max"],
        default: "min",
      },
    ],
  },
  ModelCheckpoint: {
    title: "Model Checkpoint",
    color: "#f43f5e",
    inputs: [],
    outputs: ["callback"],
    params: [
      {
        key: "dirpath",
        label: "DIR PATH",
        type: "text",
        default: "checkpoints",
      },
      { key: "monitor", label: "MONITOR", type: "text", default: "val/loss" },
      {
        key: "mode",
        label: "MODE",
        type: "select",
        options: ["min", "max"],
        default: "min",
      },
      { key: "save_top_k", label: "SAVE TOP K", type: "number", default: 1 },
      {
        key: "filename",
        label: "FILENAME",
        type: "text",
        default: "{epoch}-{val/loss:.2f}",
      },
    ],
  },
  LearningRateMonitor: {
    title: "LR Monitor",
    color: "#f43f5e",
    inputs: [],
    outputs: ["callback"],
    params: [
      {
        key: "logging_interval",
        label: "LOG INTERVAL",
        type: "select",
        options: ["step", "epoch"],
        default: "epoch",
      },
    ],
  },
  RichProgressBar: {
    title: "Rich Progress Bar",
    color: "#f43f5e",
    inputs: [],
    outputs: ["callback"],
    params: [],
  },
  CustomCallback: {
    title: "Custom Callback",
    color: "#f43f5e",
    inputs: [],
    outputs: ["callback"],
    params: [
      {
        key: "class_path",
        label: "CLASS PATH",
        type: "text",
        default: "lightning.pytorch.callbacks.MyCallback",
      },
    ],
    allowCustomParams: true,
  },
});
