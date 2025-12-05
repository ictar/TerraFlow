# TerraFlow UI 🌍

> A No-Code Visual Interface for GeoAI Fine-tuning based on [Terratorch](https://terrastackai.github.io/terratorch/stable/).

**TerraFlow UI** is a lightweight, browser-based node editor designed to democratize Geospatial AI. It allows researchers and developers to visually design model architectures, configure training pipelines, and export ready-to-run Jupyter Notebooks compatible with Google Colab.

## ✨ Features

- 🎨 Visual Workflow: Drag-and-drop interface inspired by ComfyUI. No need to write complex YAML configs manually.
- 🔌 Terratorch Native: First-class support for Terratorch components (Prithvi, Swin, HLS DataModules, Segmentation Tasks).

- 🚀 One-Click Export: Compiles your visual graph into a standardized .ipynb notebook with automatic GPU checks and dependency installation.

- 🔒 Privacy First: Runs 100% in your browser. No backend server required; your graph data never leaves your device.

- ⚡ Real-time Validation: Ensures your architecture connects correctly (e.g., Model -> Task -> Trainer) before export.

## 🚀 Getting Started

### Installation

TerraFlow is designed to be zero-dependency for the UI itself.

1. Clone the repository:
```
git clone [https://github.com/your-username/TerraFlow.git](https://github.com/your-username/TerraFlow.git)
``

2. Run: Simply open `index.html` in any modern web browser (Chrome, Edge, Firefox).

*Optional: If you want to serve it locally*:
```sh
python3 -m http.server 8000
# Open http://localhost:8000 in your browser
```

### Usage

1. Add Nodes: Right-click on the canvas to add Data Modules, Models, Tasks, or Trainers.
2. Connect: Drag from an Output port (Green) to an Input port (Blue).
3. Configure: Tweak parameters like Learning Rate, Epochs, or Backbones directly on the nodes.
4. Export: Click the "Export Notebook" button to download your training job.
5. Train: Upload the .ipynb to Google Colab or your GPU server and run!


## 🛠️ Architecture

TerraFlow acts as a Graph Compiler. It translates the Directed Acyclic Graph (DAG) you build visually into a valid Hydra/OmegaConf configuration, which is then injected into a PyTorch Lightning training script.
```
graph LR
    A[Visual Graph (React Flow)] -->|Compile| B(JSON Intermediate Representation)
    B -->|Transpile| C{Hydra/YAML Config}
    C -->|Inject| D[Jupyter Notebook (.ipynb)]
    D -->|Run| E[Terratorch Training Job]
```

## 🗺️ Roadmap
- [x] Basic Node Editor & Canvas
- [x] Export to Jupyter Notebook
- [x] Support for Segmentation Tasks
- [ ] Import Functionality: Load existing YAML configs into the graph.
- [ ] Custom Nodes: Define custom model architectures via UI.
- [ ] Live Preview: Connect to a Python backend to preview dataset chips in real-time.

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