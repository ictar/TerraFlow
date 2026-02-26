window.NODE_TYPES = window.NODE_TYPES || {};
Object.assign(window.NODE_TYPES, {
  // [TRANSFORMS] - Cyan Theme
  AlbumentationsResize: {
    title: "Resize",
    color: "#06b6d4", // Cyan 500
    inputs: [],
    outputs: ["transform"],
    params: [
      { key: "height", label: "HEIGHT", type: "number", default: 224 },
      { key: "width", label: "WIDTH", type: "number", default: 224 },
    ],
  },
  AlbumentationsHorizontalFlip: {
    title: "Horizontal Flip",
    color: "#06b6d4",
    inputs: [],
    outputs: ["transform"],
    params: [{ key: "p", label: "PROBABILITY", type: "text", default: "0.5" }],
  },
  AlbumentationsVerticalFlip: {
    title: "Vertical Flip",
    color: "#06b6d4",
    inputs: [],
    outputs: ["transform"],
    params: [{ key: "p", label: "PROBABILITY", type: "text", default: "0.5" }],
  },
  AlbumentationsD4: {
    title: "D4 Transform",
    color: "#06b6d4",
    inputs: [],
    outputs: ["transform"],
    params: [{ key: "p", label: "PROBABILITY", type: "text", default: "0.5" }],
  },
  ToTensorV2: {
    title: "To Tensor",
    color: "#06b6d4",
    inputs: [],
    outputs: ["transform"],
    params: [],
  },
  CustomTransform: {
    title: "Custom Transform",
    color: "#06b6d4",
    inputs: [],
    outputs: ["transform"],
    params: [
      {
        key: "class_path",
        label: "CLASS PATH",
        type: "text",
        default: "MyTransform",
      },
    ],
    allowCustomParams: true,
  },
});
