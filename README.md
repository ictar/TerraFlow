# TerraFlow UI 🌍

> A No-Code Visual Interface for GeoAI Fine-tuning based on [Terratorch](https://terrastackai.github.io/terratorch/stable/).

**TerraFlow UI** is a lightweight, browser-based node editor designed to democratize Geospatial AI. It allows researchers and developers to visually design model architectures, configure training pipelines, and export ready-to-run Jupyter Notebooks compatible with Google Colab.

## ✨ Features

- 🧩 **Modular Architecture**:
    - **Granular Control**: Build models component-by-component using **Backbone**, **Neck**, **Decoder**, and **Head** nodes.
    - **Model Factory**: Central hub to assemble custom architectures.
- 🔌 **Terratorch Native**: First-class support for Terratorch components:
    - **Models**: Prithvi (100M, 300M, 600M), Clay, SatMAE, ScaleMAE, Swin, ResNet, etc.
    - **Tasks**: Semantic Segmentation, Pixel-wise Regression.
    - **Inference**: Tiled Inference support with configurable crop/stride.
    - **Data**: Generic DataModules (Sen1Floods11, Landsat, Geo/Non-Geo) with custom `data_root`.
    - **Transforms**: Visual pipeline for Albumentations (Resize, Flip, ToTensor), ordered spatially.
- 🧠 **Advanced Config**:
    - **Optimization**: Configurable **LR Scheduler** (ReduceLROnPlateau, CosineAnnealingLR) and **Optimizer** (AdamW, SGD).
    - **Multiple Callbacks**: Early Stopping, Model Checkpoint, LR Monitor, Rich Progress Bar.
    - **Flexible Logging**: TensorBoard, Wandb, CSV, MLFlow.
    - **Global Settings**: Experiment naming and centralized random seed management.
- 🔄 **Re-Layout & Import**:
    - **Import YAML**: Load existing `.yaml` configuration files and reconstruct the visual graph instantly, preserving modular connections.
    - **Auto Layout**: Smart "Sink-based" organizing algorithm for perfect clean diagrams.
- 🚀 **Export**: Compiles to standard `.ipynb` notebooks or raw `.yaml` configs ready for CLI training.
- 🔒 **Privacy First**: Runs 100% in your browser. No backend server required.

## 🚀 Getting Started

### Installation

TerraFlow is valid static HTML/JS. No build step required for the UI.

1. Clone the repository:
```bash
git clone https://github.com/ictar/TerraFlow.git
```

2. Run: Simply open `index.html` in any modern web browser.

### Usage

1. **Add Nodes**: Right-click canvas to add modules. Use the organized menu (Architecture, Task, Data, Trainer).
2. **Connect**: Drag from Output ports (Green) to Input ports (Blue).
3. **Configure**:
    - Tweak node parameters.
    - Use **Global Settings** (top-right) for Experiment Name and Seed.
    - Use **Relayout** (bottom-left) to organize messy graphs.
4. **Export**: Click export buttons (top-right) for `.ipynb` or `.yaml`.

## 🛠️ Architecture

TerraFlow is now fully modularized:
- `js/nodes.js`: Node definitions (Schema).
- `js/ui.js`: Rendering engine, interaction logic, and "Sink-Based" auto-layout.
- `js/compiler.js`: Logic to transpile the graph into Hydra/Lightning configs.
- `js/importer.js`: Logic to parse YAML and reconstruct the graph.
- `js/state.js`: Global state management.

## 🗺️ Roadmap
- [x] Basic Node Editor & Canvas
- [x] Export to Jupyter Notebook & YAML
- [x] **Import Functionality**: Load existing YAMLs.
- [x] **Auto Layout**: Intelligent graph organization.
- [x] **Modular Architecture**: Backbone/Neck/Decoder/Head splitting.
- [x] **Advanced Tasks**: Regression & Tiled Inference.
- [ ] Custom Nodes: Define custom model architectures via UI.
- [ ] Live Preview: Connect to a Python backend to preview dataset chips.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
3. Commit your Changes (git commit -m 'Add some AmazingFeature')
4. Push to the Branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## License

Distributed under the MIT License. See LICENSE for more information.

Built with ❤️ for the GeoAI Community.