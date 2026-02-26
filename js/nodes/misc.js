window.NODE_TYPES = window.NODE_TYPES || {};
Object.assign(window.NODE_TYPES, {
  // [TILED INFERENCE] - Violet Theme (Darker)
  TiledInference: {
    title: "Tiled Inference",
    color: "#6d28d9", // Violet 700
    inputs: [],
    outputs: ["tiled_inference"],
    params: [
      { key: "h_crop", label: "H CROP", type: "number", default: 224 },
      { key: "w_crop", label: "W CROP", type: "number", default: 224 },
      { key: "h_stride", label: "H STRIDE", type: "number", default: 192 },
      { key: "w_stride", label: "W STRIDE", type: "number", default: 192 },
      {
        key: "average_patches",
        label: "AVG PATCHES",
        type: "select",
        options: ["True", "False"],
        default: "True",
      },
    ],
  },
});
