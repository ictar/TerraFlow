window.NODE_TYPES = window.NODE_TYPES || {};
Object.assign(window.NODE_TYPES, {
  // [OPTIMIZER] - Amber Theme
  // [OPTIMIZER & SCHEDULER] - Amber Theme
  OptimizerConfig: {
    title: "Optimizer",
    color: "#f59e0b", // Amber 500
    inputs: [],
    outputs: ["optim_args"],
    params: [
      {
        key: "type",
        label: "TYPE",
        type: "select",
        options: ["AdamW", "SGD"],
        default: "AdamW",
      },
      { key: "lr", label: "LEARNING RATE", type: "text", default: "1e-4" },
      {
        key: "weight_decay",
        label: "WEIGHT DECAY",
        type: "text",
        default: "0.05",
      },
    ],
  },
  LRScheduler: {
    title: "LR Scheduler",
    color: "#d97706", // Amber 600
    inputs: [],
    outputs: ["scheduler_args"],
    params: [
      {
        key: "type",
        label: "TYPE",
        type: "select",
        options: ["ReduceLROnPlateau", "CosineAnnealingLR"],
        default: "ReduceLROnPlateau",
      },
      {
        key: "monitor",
        label: "MONITOR (Plateau)",
        type: "text",
        default: "val/loss",
      },
      {
        key: "mode",
        label: "MODE",
        type: "select",
        options: ["min", "max"],
        default: "min",
      },
      { key: "factor", label: "FACTOR", type: "text", default: "0.1" },
      {
        key: "patience",
        label: "PATIENCE (Plateau)",
        type: "number",
        default: 10,
      },
      { key: "threshold", label: "THRESHOLD", type: "text", default: "1e-4" },
      {
        key: "threshold_mode",
        label: "THRESH MODE",
        type: "select",
        options: ["rel", "abs"],
        default: "rel",
      },
      { key: "cooldown", label: "COOLDOWN", type: "number", default: 0 },
      { key: "min_lr", label: "MIN LR", type: "text", default: "0" },
      { key: "eps", label: "EPS", type: "text", default: "1e-8" },
      { key: "t_max", label: "T_MAX (Cosine)", type: "number", default: 50 },
    ],
  },
  CustomOptimizer: {
    title: "Custom Optimizer",
    color: "#f59e0b",
    inputs: [],
    outputs: ["optim_args"],
    params: [
      {
        key: "type",
        label: "CLASS PATH",
        type: "text",
        default: "torch.optim.Adam",
      },
    ],
    allowCustomParams: true,
  },
  CustomLRScheduler: {
    title: "Custom Scheduler",
    color: "#d97706",
    inputs: [],
    outputs: ["scheduler_args"],
    params: [
      {
        key: "type",
        label: "CLASS PATH",
        type: "text",
        default: "torch.optim.lr_scheduler.StepLR",
      },
    ],
    allowCustomParams: true,
  },
});
