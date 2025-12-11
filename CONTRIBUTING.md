# Contributing to TerraFlow

First off, thank you for considering contributing to TerraFlow! It's people like you that make tools like this great.

TerraFlow is designed to be a lightweight, zero-dependency visual interface for GeoAI. We intentionally keep the architecture simple (Vanilla JS + HTML + CSS) to ensure it remains accessible and easy to hack on.

## 🚀 Getting Started

Since TerraFlow is a static web application, setting it up is incredibly easy.

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/TerraFlow.git
    cd TerraFlow
    ```
3.  **Run locally**:
    You can simply open `index.html` in your browser.
    
    For a better experience (avoiding CORS issues with local files), run a simple temporary server:
    ```bash
    # Python 3
    python3 -m http.server 8000
    
    # or Node.js
    npx http-server
    ```
    Then visit `http://localhost:8000`.

## 📂 Project Structure

- **`index.html`**: The main entry point. Contains the HTML structure, Context Menu, and inline SVG layers.
- **`css/style.css`**: All styling (Nodes, Canvas, UI Overlay). We use standard CSS variables and simple classes.
- **`js/`**:
    - **`app.js`**: Application entry point and initialization logic.
    - **`nodes.js`**: The Schema Definition. Add new Node Types, Parameters, and inputs/outputs here.
    - **`ui.js`**: The Rendering Engine. Handles Canvas interactions, Node Rendering, Drag-and-Drop, and Auto-Layout logic.
    - **`compiler.js`**: Logic to transpile the Visual Graph into `Hydra/YAML` configs and `Jupyter Notebooks`.
    - **`importer.js`**: Logic to parse YAML files and reconstruct the Visual Graph.
    - **`state.js`**: Simple global state management (current nodes, list of connections, pan/zoom level).

## 🛠️ How to Add a New Feature

### Adding a New Node Type
1.  Open `js/nodes.js`.
2.  Add a new entry to the `NODE_TYPES` object.
3.  Define its `title`, `color`, `inputs`, `outputs`, and `params`.
4.  Open `index.html` and add a corresponding entry to the `#context-menu` to make it accessible.

### Improving the Compiler
1.  Open `js/compiler.js`.
2.  Modify `buildHydraConfig()` or `buildCLIConfig()` to handle the new node data and map it to the correct Terratorch/Lightning configuration structure.

## 🎨 Style Guide

- **JavaScript**: We use modern Vanilla JavaScript (ES6+). No TypeScript, no React/Vue, no Build steps. Keep it simple.
- **CSS**: Plain CSS. Tailwind classes are used in `index.html` for utility, but core graph styling resides in `css/style.css`.
- **Commits**: Please write clear, descriptive commit messages.

## 🐛 Reporting Bugs

Bugs are tracked as GitHub Issues. Create an issue and provide as much detail as possible, including:
- Steps to reproduce the bug.
- Browser version and OS.
- Any errors seen in the Developer Console (F12).

## 📥 Submitting a Pull Request

1.  Create a new branch for your feature: `git checkout -b feature/amazing-feature`.
2.  Commit your changes.
3.  Push to the branch.
4.  Open a Pull Request against the `main` branch of the original repository.

We look forward to your contributions! 🌍
