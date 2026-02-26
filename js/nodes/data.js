window.NODE_TYPES = window.NODE_TYPES || {};
Object.assign(window.NODE_TYPES, {
  // [DATA MODULE] - Emerald Theme
  // [DATA MODULE] - Emerald Theme
  // [DATA MODULE] - Emerald Theme
  DataModule: {
    title: "Data Module",
    color: "#10b981", // Emerald 500
    inputs: ["train_transform", "val_transform", "test_transform"],
    outputs: ["datamodule"],
    params: [
      {
        key: "class_path",
        label: "CLASS PATH",
        type: "select",
        options: [
          "GenericNonGeoPixelwiseRegressionDataModule",
          "GenericNonGeoSegmentationDataModule",
          "Sen1Floods11NonGeoDataModule",
          "Landsat7DataModule",
          "TorchGeoDataModule",
        ],
        default: "GenericNonGeoSegmentationDataModule",
      },
      { key: "batch_size", label: "BATCH SIZE", type: "number", default: 4 },
      { key: "num_workers", label: "WORKERS", type: "number", default: 4 },
      {
        key: "train_data_root",
        label: "TRAIN ROOT",
        type: "text",
        default: "data/train",
      },
      {
        key: "val_data_root",
        label: "VAL ROOT",
        type: "text",
        default: "data/val",
      },
      {
        key: "test_data_root",
        label: "TEST ROOT",
        type: "text",
        default: "data/test",
      },
      {
        key: "means",
        label: "MEANS",
        type: "text",
        default: "[0.485, 0.456, 0.406]",
      },
      {
        key: "stds",
        label: "STDS",
        type: "text",
        default: "[0.229, 0.224, 0.225]",
      },
      { key: "num_classes", label: "NUM CLASSES", type: "number", default: 1 },
      { key: "img_grep", label: "IMG GREP", type: "text", default: "*" },
      { key: "label_grep", label: "LABEL GREP", type: "text", default: "*" },
      {
        key: "bands",
        label: "BANDS",
        type: "text",
        default: "BLUE,GREEN,RED,NIR",
      },
      {
        key: "rgb_indices",
        label: "RGB INDICES",
        type: "text",
        default: "0,1,2",
      },
      {
        key: "reduce_zero_label",
        label: "REDUCE ZERO LBL",
        type: "select",
        options: ["True", "False"],
        default: "False",
      },
      {
        key: "constant_scale",
        label: "CONST SCALE",
        type: "text",
        default: "0.0001",
      },
      {
        key: "no_data_replace",
        label: "NO DATA VAL",
        type: "number",
        default: 0,
      },
      {
        key: "no_label_replace",
        label: "NO LABEL VAL",
        type: "number",
        default: -1,
      },
      {
        key: "drop_last",
        label: "DROP LAST",
        type: "select",
        options: ["True", "False"],
        default: "True",
      },
    ],
  },
  CustomDataModule: {
    title: "Custom DataModule",
    color: "#10b981",
    inputs: ["train_transform", "val_transform", "test_transform"],
    outputs: ["datamodule"],
    params: [
      {
        key: "class_path",
        label: "CLASS PATH",
        type: "text",
        default: "MyDataModule",
      },
    ],
    allowCustomParams: true,
  },
});
