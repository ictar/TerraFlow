window.NODE_TYPES = window.NODE_TYPES || {};
Object.assign(window.NODE_TYPES, {
  // [LOGGER] - Pink Theme
  // [LOGGER] - Pink Theme
  Logger: {
    title: "Logger",
    color: "#ec4899", // Pink 500
    inputs: [],
    outputs: ["logger"],
    params: [
      {
        key: "type",
        label: "TYPE",
        type: "select",
        options: ["TensorBoard", "Wandb", "CSV", "MLFlow"],
        default: "TensorBoard",
      },
      {
        key: "project",
        label: "PROJECT (Wandb/MLFlow)",
        type: "text",
        default: "terraflow_project",
      },
      {
        key: "name",
        label: "RUN NAME",
        type: "text",
        default: "run_version_1",
      },
      { key: "save_dir", label: "SAVE DIR", type: "text", default: "logs" },
    ],
  },
  CustomLogger: {
    title: "Custom Logger",
    color: "#ec4899",
    inputs: [],
    outputs: ["logger"],
    params: [
      {
        key: "type",
        label: "CLASS PATH",
        type: "text",
        default: "lightning.pytorch.loggers.WandbLogger",
      },
    ],
    allowCustomParams: true,
  },
});
