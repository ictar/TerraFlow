window.NODE_TYPES = window.NODE_TYPES || {};
Object.assign(window.NODE_TYPES, {
  // [TRAINER] - Blue Theme
  TrainerConfig: {
    title: "Lightning Trainer",
    color: "#3b82f6", // Blue 500
    inputs: ["datamodule", "task", "logger", "callbacks"],
    outputs: [],
    params: [
      { key: "max_epochs", label: "MAX EPOCHS", type: "number", default: 50 },
      {
        key: "accelerator",
        label: "ACCELERATOR",
        type: "select",
        options: ["gpu", "cpu", "auto"],
        default: "auto",
      },
      { key: "devices", label: "DEVICES", type: "text", default: "auto" },
      { key: "strategy", label: "STRATEGY", type: "text", default: "auto" },
      { key: "num_nodes", label: "NUM NODES", type: "number", default: 1 },
      {
        key: "precision",
        label: "PRECISION",
        type: "select",
        options: ["16-mixed", "32"],
        default: "16-mixed",
      },
      {
        key: "check_val_every_n_epoch",
        label: "VAL CHECK INTERVAL",
        type: "number",
        default: 2,
      },
      {
        key: "log_every_n_steps",
        label: "LOG STEP INTERVAL",
        type: "number",
        default: 10,
      },
      {
        key: "enable_checkpointing",
        label: "ENABLE CHECKPOINTING",
        type: "select",
        options: ["True", "False"],
        default: "True",
      },
      {
        key: "default_root_dir",
        label: "ROOT DIR",
        type: "text",
        default: "checkpoints",
      },
    ],
  },
});
