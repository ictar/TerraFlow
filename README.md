# TerraFlow UI 🌍

> A No-Code Visual Interface for GeoAI Fine-tuning based on [Terratorch](https://terrastackai.github.io/terratorch/stable/).

**TerraFlow UI** is a lightweight, browser-based node editor designed to democratize Geospatial AI. It allows researchers and developers to visually design model architectures, configure training pipelines, and export ready-to-run Jupyter Notebooks compatible with Google Colab.

## ✨ Features

- 🎨 **Visual Workflow**: Drag-and-drop interface inspired by ComfyUI/SVP. Modular CSS/JS structure for easy maintenance.
- 🔌 **Terratorch Native**: First-class support for Terratorch components:
    - **Models**: Prithvi (100M, 300M), Clay, SatMAE, ScaleMAE, Swin, ResNet, etc.
    - **Tasks**: Segmentation, Pixel-wise Regression.
    - **Data**: HLS DataModule (customizable bands).
- 🧠 **Advanced Config**:
    - **Multiple Callbacks**: Early Stopping, Model Checkpoint, LR Monitor, Rich Progress Bar.
    - **Flexible Logging**: TensorBoard, Wandb, CSV, MLFlow.
    - **Global Settings**: Centralized management for random seeds and environment configs.
- 🔄 **Re-Layout & Import**:
    - **Import YAML**: Load existing `.yaml` configuration files and reconstruct the visual graph instantly.
    - **Auto Layout**: One-click graph organization to untangle complex pipelines.
- 🚀 **Export**: Compiles to standardized `.ipynb` notebooks or raw `.yaml` configs.
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

1. **Add Nodes**: Right-click canvas to add modules (Data, Models, Tasks) or specialized components (Loggers, Callbacks).
2. **Connect**: Drag from Output ports (Green) to Input ports (Blue).
3. **Configure**:
    - Tweak node parameters (Learning Rate, Backbones).
    - Use the **Global Settings** panel (bottom-right) for Seed/Checkpoint config.
    - Use the **Relayout** button (bottom-left) to organize messy graphs.
4. **Export**: Click export buttons (top-right) for `.ipynb` or `.yaml`.

## 🛠️ Architecture

TerraFlow is now fully modularized:
- `js/nodes.js`: Node definitions and schema.
- `js/ui.js`: Rendering engine, interaction logic, and auto-layout.
- `js/compiler.js`: Logic to transpile the graph into Hydra/Lightning configs.
- `js/importer.js`: Logic to parse YAML and reconstruct the graph.
- `js/state.js`: Global state management.

## 🗺️ Roadmap
- [x] Basic Node Editor & Canvas
- [x] Export to Jupyter Notebook & YAML
- [x] **Import Functionality**: Load existing YAMLs.
- [x] **Auto Layout**: Intelligent graph organization.
- [x] **Advanced Callbacks & Loggers**: Full Lightning support.
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