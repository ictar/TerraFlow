window.NODE_TYPES = window.NODE_TYPES || {};
Object.assign(window.NODE_TYPES, {
  // [ARCHITECTURE] - Violet Theme
  ModelBackbone: {
    title: "Backbone",
    color: "#7c3aed", // Violet 600
    inputs: [],
    outputs: ["backbone_config"],
    params: [
      {
        key: "model_name",
        label: "MODEL",
        type: "select",
        options: [
          "prithvi_eo_v1_100",
          "prithvi_eo_v2_300",
          "prithvi_eo_v2_600",
          "clay_v1_base",
          "satmae_vit_base_patch16",
          "scale_mae",
          "swin_base_patch4_window12_384",
          "resnet50",
          "dofa_base_patch16_224",
          "terramind_v1_base",
        ],
        default: "prithvi_eo_v1_100",
      },
      {
        key: "pretrained",
        label: "PRETRAINED",
        type: "select",
        options: ["True", "False"],
        default: "True",
      },
      { key: "in_channels", label: "IN CHANNELS", type: "number", default: 6 },
      { key: "num_frames", label: "NUM FRAMES", type: "number", default: 1 },
      {
        key: "backbone_modalities",
        label: "MODALITIES",
        type: "text",
        default: "",
      },
      {
        key: "drop_path_rate",
        label: "DROP PATH RATE",
        type: "text",
        default: "0.3",
      },
      { key: "window_size", label: "WINDOW SIZE", type: "number", default: 8 },
    ],
  },
  ModelDecoder: {
    title: "Decoder",
    color: "#8b5cf6", // Violet 500
    inputs: [],
    outputs: ["decoder_config"],
    params: [
      {
        key: "decoder_name",
        label: "DECODER",
        type: "select",
        options: ["UperNetDecoder", "FCNDecoder", "IdentityDecoder"],
        default: "UperNetDecoder",
      },
      {
        key: "decoder_channels",
        label: "CHANNELS",
        type: "number",
        default: 256,
      },
      {
        key: "scale_modules",
        label: "SCALE MODULES",
        type: "select",
        options: ["True", "False"],
        default: "True",
      },
    ],
  },
  ModelHead: {
    title: "Task Head",
    color: "#a78bfa", // Violet 400
    inputs: [],
    outputs: ["head_config"],
    params: [
      { key: "dropout", label: "DROPOUT", type: "text", default: "0.1" },
      {
        key: "learned_upscale_layers",
        label: "UPSCALE LAYERS",
        type: "number",
        default: 1,
      },
      {
        key: "final_act",
        label: "FINAL ACTIVATION",
        type: "select",
        options: ["None", "ReLU", "Sigmoid", "Tanh"],
        default: "None",
      },
    ],
  },
  ModelNeck: {
    title: "Model Neck",
    color: "#8b5cf6", // Violet 500
    inputs: [],
    outputs: ["neck_config"],
    params: [
      {
        key: "indices",
        label: "INDICES (comma-sep)",
        type: "text",
        default: "2,5,8,11",
      },
      {
        key: "reshape",
        label: "RESHAPE TOKENS",
        type: "select",
        options: ["True", "False"],
        default: "True",
      },
    ],
  },
  ModelFactory: {
    title: "Model Factory",
    color: "#5b21b6", // Violet 800 (Darker main node)
    inputs: ["backbone", "decoder", "head", "neck"],
    outputs: ["model_args"],
    params: [
      // Keeps simple default behavior if nothing connected, or simple params?
      // To enforce modularity, let's keep params empty or minimal.
      // But for backward compatibility/simplicity, maybe provide defaults?
      // Let's assume input-driven for now.
    ],
  },
  CustomBackbone: {
    title: "Custom Backbone",
    color: "#7c3aed",
    inputs: [],
    outputs: ["backbone_config"],
    params: [
      {
        key: "model_name",
        label: "MODEL NAME",
        type: "text",
        default: "my_backbone",
      },
    ],
    allowCustomParams: true,
  },
  CustomDecoder: {
    title: "Custom Decoder",
    color: "#8b5cf6",
    inputs: [],
    outputs: ["decoder_config"],
    params: [
      {
        key: "decoder_name",
        label: "DECODER NAME",
        type: "text",
        default: "MyDecoder",
      },
    ],
    allowCustomParams: true,
  },
  CustomHead: {
    title: "Custom Head",
    color: "#a78bfa",
    inputs: [],
    outputs: ["head_config"],
    params: [],
    allowCustomParams: true,
  },
  CustomNeck: {
    title: "Custom Neck",
    color: "#8b5cf6",
    inputs: [],
    outputs: ["neck_config"],
    params: [],
    allowCustomParams: true,
  },
});
