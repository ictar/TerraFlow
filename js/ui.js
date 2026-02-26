/* UI Logic: Rendering, Interactions, and Node Management */

function render() {
  nodesContainer.style.transform = `scale(${state.zoom}) translate(${state.pan.x}px, ${state.pan.y}px)`;
  // svgLayer is a child of nodesContainer, so it inherits the transform automatically.
  // Applying it again causes double-scaling/translation.
  renderConnections();
}

// --- DOM Creation ---

function createNodeDOM(node) {
  const el = document.createElement("div");
  el.className = `node ${node.selected ? "selected" : ""}`;
  el.id = node.id;
  el.style.left = node.x + "px";
  el.style.top = node.y + "px";
  el.style.borderColor = NODE_TYPES[node.type].color;

  // Header
  const header = document.createElement("div");
  header.className = "node-header";
  header.style.background = NODE_TYPES[node.type].color + "44"; // Low opacity bg
  header.innerHTML = `
        <span class="flex items-center gap-2">
            <span class="badge" style="background:${NODE_TYPES[node.type].color}"></span>
            ${NODE_TYPES[node.type].title}
        </span>
        <i class="ph-bold ph-x text-gray-400 hover:text-white cursor-pointer" onclick="deleteNode('${node.id}')"></i>
    `;
  header.onmousedown = (e) => {
    state.isDraggingNode = node;
    state.dragStart = { x: e.clientX, y: e.clientY };
    e.stopPropagation();

    // Selection Logic
    state.nodes.forEach((n) => (n.selected = false));
    node.selected = true;
    document
      .querySelectorAll(".node")
      .forEach((n) => n.classList.remove("selected"));
    el.classList.add("selected");
  };
  el.appendChild(header);

  // Body
  const body = document.createElement("div");
  body.className = "node-body";

  // Inputs
  NODE_TYPES[node.type].inputs.forEach((inputKey) => {
    const row = document.createElement("div");
    row.className = "socket-row input";
    row.innerHTML = `<span class="socket-label">${formatPortName(inputKey)}</span>`;

    const socket = document.createElement("div");
    socket.className = "port input";
    socket.dataset.port = inputKey;
    socket.dataset.node = node.id;
    socket.dataset.type = "input";

    // Wire connect logic
    socket.onmousedown = (e) => startLineDrag(e, node.id, inputKey, "input");

    row.prepend(socket);
    body.appendChild(row);
  });

  // Params
  NODE_TYPES[node.type].params.forEach((param) => {
    const wrapper = document.createElement("div");
    wrapper.className = "node-input-wrapper";
    wrapper.innerHTML = `<span class="node-input-label">${param.label}</span>`;

    let input;
    if (param.type === "select") {
      input = document.createElement("select");
      input.className = "node-input";
      input.id = `${node.id}_${param.key}`;
      input.name = `${node.id}_${param.key}`;
      input.setAttribute("autocomplete", "off"); // Prevent autofill suggestions
      let foundSelected = false;
      param.options.forEach((opt) => {
        const optEl = document.createElement("option");
        optEl.value = opt;
        optEl.innerText = opt;
        if (opt === node.data[param.key]) {
          optEl.selected = true;
          foundSelected = true;
        }
        input.appendChild(optEl);
      });

      // If the current value is not in the predefined options, add it dynamically
      const currVal = node.data[param.key];
      if (
        !foundSelected &&
        currVal !== undefined &&
        currVal !== null &&
        currVal !== ""
      ) {
        const customOpt = document.createElement("option");
        customOpt.value = currVal;
        customOpt.innerText = currVal;
        customOpt.selected = true;
        input.appendChild(customOpt);
      }
    } else {
      input = document.createElement("input");
      input.className = "node-input";
      input.type = param.type === "number" ? "number" : "text";
      input.value = node.data[param.key];
      input.id = `${node.id}_${param.key}`;
      input.name = `${node.id}_${param.key}`;
      input.setAttribute("autocomplete", "off"); // Prevent autofill suggestions
    }

    input.oninput = (e) => {
      node.data[param.key] = e.target.value;
    };
    // Prevent pan when typing
    input.onmousedown = (e) => e.stopPropagation();

    wrapper.appendChild(input);
    body.appendChild(wrapper);
  });

  // Custom Params Support
  if (NODE_TYPES[node.type].allowCustomParams) {
    if (!node.data.customParams) node.data.customParams = [];

    const customContainer = document.createElement("div");
    customContainer.className =
      "custom-params-container flex flex-col gap-2 mt-2";

    const renderCustomParams = () => {
      customContainer.innerHTML = "";
      node.data.customParams.forEach((param, index) => {
        const wrapper = document.createElement("div");
        wrapper.className =
          "border border-gray-700 bg-gray-800/50 p-2 rounded relative flex flex-col gap-1";

        // Header (Param #) + Delete
        const header = document.createElement("div");
        header.className = "flex justify-between items-center";
        const label = document.createElement("span");
        label.className =
          "text-[10px] uppercase font-bold text-gray-500 tracking-wider";
        label.innerText = `Param ${index + 1}`;

        const delBtn = document.createElement("i");
        delBtn.className =
          "ph-bold ph-trash text-red-400 cursor-pointer hover:text-red-300 text-xs";
        delBtn.onmousedown = (e) => e.stopPropagation(); // prevent drag
        delBtn.onclick = () => {
          node.data.customParams.splice(index, 1);
          renderCustomParams();
        };

        header.appendChild(label);
        header.appendChild(delBtn);
        wrapper.appendChild(header);

        // Key Input
        const keyInput = document.createElement("input");
        keyInput.className =
          "w-full bg-[#121212] border border-[#333] rounded px-1.5 py-1 text-xs text-blue-300 font-mono focus:border-blue-500 outline-none transition";
        keyInput.placeholder = "key (e.g. lr)";
        keyInput.value = param.key;
        keyInput.oninput = (e) => {
          param.key = e.target.value;
        };
        keyInput.onmousedown = (e) => e.stopPropagation();
        wrapper.appendChild(keyInput);

        // Value Input
        const valInput = document.createElement("input");
        valInput.className =
          "w-full bg-[#121212] border border-[#333] rounded px-1.5 py-1 text-xs text-green-300 font-mono focus:border-green-500 outline-none transition";
        valInput.placeholder = "value";
        valInput.value = param.value;
        valInput.oninput = (e) => {
          param.value = e.target.value;
        };
        valInput.onmousedown = (e) => e.stopPropagation();
        wrapper.appendChild(valInput);

        customContainer.appendChild(wrapper);
      });

      // Add Params Button
      const addBtn = document.createElement("button");
      addBtn.className =
        "w-full py-1.5 text-[10px] font-bold uppercase tracking-wider bg-gray-700 hover:bg-gray-600 rounded text-gray-300 transition flex items-center justify-center gap-1";
      addBtn.innerHTML = '<i class="ph-bold ph-plus"></i> Add Param';
      addBtn.onclick = () => {
        node.data.customParams.push({ key: "", value: "" });
        renderCustomParams();
      };
      addBtn.onmousedown = (e) => e.stopPropagation();
      customContainer.appendChild(addBtn);
    };

    renderCustomParams();
    body.appendChild(customContainer);
  }

  // Outputs
  NODE_TYPES[node.type].outputs.forEach((outputKey) => {
    const row = document.createElement("div");
    row.className = "socket-row output";
    row.innerHTML = `<span class="socket-label">${formatPortName(outputKey)}</span>`;

    const socket = document.createElement("div");
    socket.className = "port output";
    socket.dataset.port = outputKey;
    socket.dataset.node = node.id;
    socket.dataset.type = "output";

    // Wire connect logic
    socket.onmousedown = (e) => startLineDrag(e, node.id, outputKey, "output");

    row.appendChild(socket);
    body.appendChild(row);
  });

  el.appendChild(body);
  nodesContainer.appendChild(el);
}

function formatPortName(key) {
  return key.replace(/_/g, " ").toUpperCase();
}

// --- Connections ---

// function renderConnections() { ... } // Removed duplicate definition here. Using the one defined at the bottom.

function drawCurve(p1, p2, isTemp, connData = null, orderIndex = null) {
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  // Dynamic Curvature:
  const deltaX = Math.abs(p2.x - p1.x);
  const dist = Math.max(deltaX * 0.5, 60);

  // Wire flow: always out to the right (p1 + dist) and in from the left (p2 - dist)
  const cp1 = { x: p1.x + dist, y: p1.y };
  const cp2 = { x: p2.x - dist, y: p2.y };

  // Improve vertical handling: If going backwards (p2.x < p1.x), loop around
  if (p2.x < p1.x + 20) {
    cp1.x = Math.max(p1.x + 200, p1.x + deltaX);
    cp2.x = Math.min(p2.x - 200, p2.x - deltaX);
  }

  const d = `M ${p1.x} ${p1.y} C ${cp1.x} ${cp1.y} ${cp2.x} ${cp2.y} ${p2.x} ${p2.y}`;

  path.setAttribute("d", d);
  path.setAttribute("class", isTemp ? "drag-line" : "connection");

  // Colorize based on Source Node
  let sourceNodeId = null;
  if (isTemp && state.tempLine) {
    sourceNodeId = state.tempLine.startNode;
  } else if (connData) {
    sourceNodeId = connData.sourceNode;
  }

  let strokeColor = null;
  if (sourceNodeId) {
    const node = state.nodes.find((n) => n.id === sourceNodeId);
    if (node && NODE_TYPES[node.type]) {
      strokeColor = NODE_TYPES[node.type].color;
      path.style.stroke = strokeColor;
    }
  }

  if (!isTemp && connData) {
    path.oncontextmenu = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (confirm("Delete this connection?")) {
        deleteConnection(connData);
      }
    };
  }

  svgLayer.appendChild(path);

  // Render Order Badge (if applicable)
  if (orderIndex !== null) {
    // Calculate Bezier midpoint (t = 0.5)
    // B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    const t = 0.5;
    const mt = 1 - t; // 0.5
    const mt2 = mt * mt; // 0.25
    const mt3 = mt2 * mt; // 0.125
    const t2 = t * t; // 0.25
    const t3 = t2 * t; // 0.125

    const midX =
      mt3 * p1.x + 3 * mt2 * t * cp1.x + 3 * mt * t2 * cp2.x + t3 * p2.x;
    const midY =
      mt3 * p1.y + 3 * mt2 * t * cp1.y + 3 * mt * t2 * cp2.y + t3 * p2.y;

    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("transform", `translate(${midX}, ${midY})`);

    const circle = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "circle",
    );
    circle.setAttribute("r", "10");
    circle.setAttribute("fill", strokeColor || "#555");
    circle.setAttribute("stroke", "#121212");
    circle.setAttribute("stroke-width", "2");

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("dy", "3"); // Vertical center adjustment
    text.setAttribute("fill", "#fff");
    text.setAttribute("font-size", "10px");
    text.setAttribute("font-weight", "bold");
    text.textContent = orderIndex;

    g.appendChild(circle);
    g.appendChild(text);
    g.style.pointerEvents = "none"; // Click through to wire

    svgLayer.appendChild(g);
  }
}

function getPortPosition(nodeId, portName, type) {
  const nodeEl = document.getElementById(nodeId);
  if (!nodeEl) return null;
  const portEl = nodeEl.querySelector(
    `.port[data-port="${portName}"][data-type="${type}"]`,
  );
  if (!portEl) return null;

  // Reliable approach: Use getBoundingClientRect delta, scaled by zoom
  const portRect = portEl.getBoundingClientRect();
  const containerRect = nodesContainer.getBoundingClientRect();

  // Calculate relative position within the zoomed/panned container context
  const x =
    (portRect.left - containerRect.left) / state.zoom +
    portRect.width / 2 / state.zoom;
  const y =
    (portRect.top - containerRect.top) / state.zoom +
    portRect.height / 2 / state.zoom;

  return { x, y };
}

// --- Coordinates Helper ---

function transformMouseToCanvas(clientX, clientY) {
  const rect = nodesContainer.getBoundingClientRect();
  return {
    x: (clientX - rect.left) / state.zoom,
    y: (clientY - rect.top) / state.zoom,
  };
}

// --- Interactions ---

function startLineDrag(e, nodeId, portName, type) {
  e.stopPropagation();
  e.preventDefault();
  const startPos = getPortPosition(nodeId, portName, type);
  state.tempLine = {
    startNode: nodeId,
    startPort: portName,
    type: type,
    start: startPos,
    curr: startPos,
  };
}

function deleteNode(id) {
  if (!confirm("Delete node?")) return;
  state.nodes = state.nodes.filter((n) => n.id !== id);
  state.connections = state.connections.filter(
    (c) => c.sourceNode !== id && c.targetNode !== id,
  );

  // DOM cleanup
  const el = document.getElementById(id);
  if (el) el.remove();

  renderConnections();
}

function deleteConnection(conn) {
  state.connections = state.connections.filter((c) => c !== conn);
  renderConnections();
}

// --- Global Actions for HTML Buttons ---

function addNode(type, x, y, initialData = {}) {
  // Use a random suffix to ensure uniqueness even if called rapidly
  const id = "node_" + Date.now() + "_" + Math.floor(Math.random() * 1000);
  const newNode = {
    id,
    type,
    x: x || 100,
    y: y || 100,
    data: { ...initialData },
  };

  // Fill missing params with defaults
  NODE_TYPES[type].params.forEach((p) => {
    if (newNode.data[p.key] === undefined) newNode.data[p.key] = p.default;
  });

  state.nodes.push(newNode);
  createNodeDOM(newNode);
  contextMenu.style.display = "none";
  return newNode;
}

function clearCanvas() {
  if (!confirm("Clear all?")) return;
  state.nodes = [];
  state.connections = [];
  nodesContainer.innerHTML = "";
  nodesContainer.appendChild(svgLayer); // Restore SVG
  svgLayer.innerHTML = "";
}

// Fit to Screen / Reset View
function resetView() {
  if (state.nodes.length === 0) {
    state.pan = { x: 0, y: 0 };
    state.zoom = 1;
    render();
    return;
  }

  // Calculate Bounding Box
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  state.nodes.forEach((n) => {
    if (n.x < minX) minX = n.x;
    if (n.y < minY) minY = n.y;
    // Approx width/height - assuming 200x100 for safety or using DOM if available
    // Using a safe estimate
    const width = 250;
    const height = 150;
    if (n.x + width > maxX) maxX = n.x + width;
    if (n.y + height > maxY) maxY = n.y + height;
  });

  // Add padding
  const PADDING = 50;
  minX -= PADDING;
  minY -= PADDING;
  maxX += PADDING;
  maxY += PADDING;

  const graphW = maxX - minX;
  const graphH = maxY - minY;

  // Viewport size
  const container = document.getElementById("nodes-container").parentElement; // #canvas-bg or wrapper
  const viewW = container.clientWidth || window.innerWidth;
  const viewH = container.clientHeight || window.innerHeight;

  // Determine scale to fit
  const scaleX = viewW / graphW;
  const scaleY = viewH / graphH;
  let newScale = Math.min(scaleX, scaleY);

  // Clamp scale
  newScale = Math.min(Math.max(newScale, 0.2), 1.2);

  state.zoom = newScale;

  // Center
  // center of graph = minX + graphW/2
  // center of view = viewW/2
  // pan = viewCenter - (graphCenter * scale)

  state.pan.x = viewW / 2 - (minX + graphW / 2) * newScale;
  state.pan.y = viewH / 2 - (minY + graphH / 2) * newScale;

  render();
}
window.resetView = resetView; // Expose to global scope for HTML button
window.addNode = addNode; // Ensure other globals are exposed if needed
window.autoLayout = autoLayout;
window.clearCanvas = clearCanvas;

function autoLayout() {
  // 1. Build Adjacency & Reverse Adjacency
  const adj = new Map(); // ID -> [TargetIDs]
  const revAdj = new Map(); // ID -> [SourceIDs]
  const outDegree = new Map();

  state.nodes.forEach((node) => {
    adj.set(node.id, []);
    revAdj.set(node.id, []);
    outDegree.set(node.id, 0);
  });

  state.connections.forEach((conn) => {
    adj.get(conn.sourceNode).push(conn.targetNode);
    revAdj.get(conn.targetNode).push(conn.sourceNode);
    outDegree.set(conn.sourceNode, outDegree.get(conn.sourceNode) + 1);
  });

  // 2. Compute Depth from Sinks (Longest Path to Sink)
  // Sinks are nodes with outDegree 0 (usually Trainer)
  const depthFromSink = new Map();

  // Initialize depths to -1
  state.nodes.forEach((n) => depthFromSink.set(n.id, -1));

  function getDepth(nodeId, visited = new Set()) {
    if (visited.has(nodeId)) return 0; // Cycle detected or visited
    if (depthFromSink.get(nodeId) !== -1) return depthFromSink.get(nodeId);

    visited.add(nodeId);

    const targets = adj.get(nodeId);
    if (targets.length === 0) {
      depthFromSink.set(nodeId, 0);
      visited.delete(nodeId);
      return 0;
    }

    let maxChildDepth = 0;
    targets.forEach((tId) => {
      maxChildDepth = Math.max(maxChildDepth, getDepth(tId, visited));
    });

    const d = maxChildDepth + 1;
    depthFromSink.set(nodeId, d);
    visited.delete(nodeId);
    return d;
  }

  let maxGraphDepth = 0;
  state.nodes.forEach((n) => {
    const d = getDepth(n.id);
    if (d > maxGraphDepth) maxGraphDepth = d;
  });

  // 3. Group Nodes by Column (Forward Level)
  // Column = MaxDepth - DepthFromSink
  // This aligns sinks to the right, and pushes inputs to the left based on how many steps away they are.
  const columns = new Map();
  state.nodes.forEach((n) => {
    const col = maxGraphDepth - depthFromSink.get(n.id);
    if (!columns.has(col)) columns.set(col, []);
    columns.get(col).push(n.id);
  });

  // 4. Assign Grid Positions
  const START_X = 50;
  const START_Y = 80;
  const COL_WIDTH = 340;
  const NODE_GAP = 30;

  // Y-Sorting Priority (Architectural Tiering)
  const TYPE_PRIORITY = {
    // Tier 1: Data Pipeline (Top)
    // Transforms feed into DataModule, so grouping them high up keeps the Data chain at the top.
    DataModule: 1,
    AlbumentationsResize: 2,
    AlbumentationsHorizontalFlip: 2,
    AlbumentationsVerticalFlip: 2,
    AlbumentationsD4: 2,
    ToTensorV2: 2,
    CustomTransform: 2,

    // Tier 2: Model Architecture (Middle)
    ModelBackbone: 10,
    ModelNeck: 11,
    ModelDecoder: 12,
    ModelHead: 13,
    ModelFactory: 14,

    // Tier 3: Task & Optimization (Bottom)
    // These take Model inputs and feed into Trainer
    TiledInference: 20,
    OptimizerConfig: 21,
    LRScheduler: 21,
    SegmentationTask: 25,
    PixelwiseRegressionTask: 25,

    // Tier 4: Trainer & Infrastructure (Bottom-Right Sink)
    TrainerConfig: 50,
    Logger: 51,
    CustomLogger: 51,
    EarlyStopping: 52,
    ModelCheckpoint: 52,
    LearningRateMonitor: 52,
    RichProgressBar: 52,
    CustomCallback: 52,
  };

  const sortedCols = Array.from(columns.keys()).sort((a, b) => a - b);

  sortedCols.forEach((colIdx) => {
    const nodeIds = columns.get(colIdx);

    // Sort by Type Priority then ID
    nodeIds.sort((a, b) => {
      const nA = state.nodes.find((n) => n.id === a);
      const nB = state.nodes.find((n) => n.id === b);
      const pA = TYPE_PRIORITY[nA.type] || 99;
      const pB = TYPE_PRIORITY[nB.type] || 99;
      if (pA !== pB) return pA - pB;
      // Maintain previous Y-order to ensure lists (like loggers and callbacks) preserve their sequence
      if (nA.y !== nB.y) return nA.y - nB.y;
      if (nA.x !== nB.x) return nA.x - nB.x;
      return nA.id.localeCompare(nB.id);
    });

    let currentY = START_Y;

    nodeIds.forEach((id) => {
      const node = state.nodes.find((n) => n.id === id);
      if (!node) return;

      // Set Position
      node.x = START_X + colIdx * COL_WIDTH;
      node.y = currentY;

      // DOM Update
      const el = document.getElementById(node.id);
      if (el) {
        el.style.left = node.x + "px";
        el.style.top = node.y + "px";
      }

      // Estimate Node Height
      const typeDef = NODE_TYPES[node.type] || {};
      const paramCount = (typeDef.params || []).length;
      const inputCount = (typeDef.inputs || []).length;
      const outputCount = (typeDef.outputs || []).length;

      // Refined estimation
      // Header: 40, Params: ~54 (with label), Sockets: ~28
      // Compact mode calculation
      const estimatedHeight =
        44 + paramCount * 54 + inputCount * 28 + outputCount * 28;

      currentY += estimatedHeight + NODE_GAP;
    });
  });

  renderConnections();
  state.zoom = 1;
  state.pan = { x: 0, y: 0 }; // Center view roughly
  render();
}

// --- Global Settings Logic ---

function updateGlobalSetting(key, value) {
  if (key === "seed_everything") {
    state.globalConfig.seed_everything = parseInt(value) || 0;
  } else if (key === "experiment_name") {
    state.globalConfig.experiment_name = value;
  }
}

function updateGlobalUI() {
  const seedEl = document.getElementById("global-seed");
  if (seedEl) {
    seedEl.value = state.globalConfig.seed_everything;
  }
  const nameEl = document.getElementById("global-experiment-name");
  if (nameEl) {
    nameEl.value = state.globalConfig.experiment_name || "my_experiment";
  }
}

// --- Event Listeners Initialization ---

function initUIEvents() {
  // Mouse Move (Dragging Node / Pan / Wire)
  window.onmousemove = (e) => {
    if (state.isDraggingNode) {
      const dx = (e.clientX - state.dragStart.x) / state.zoom;
      const dy = (e.clientY - state.dragStart.y) / state.zoom;

      state.isDraggingNode.x += dx;
      state.isDraggingNode.y += dy;

      const el = document.getElementById(state.isDraggingNode.id);
      el.style.left = state.isDraggingNode.x + "px";
      el.style.top = state.isDraggingNode.y + "px";

      state.dragStart = { x: e.clientX, y: e.clientY };
      renderConnections();
    } else if (state.isDraggingCanvas) {
      state.pan.x += e.clientX - state.dragStart.x;
      state.pan.y += e.clientY - state.dragStart.y;
      state.dragStart = { x: e.clientX, y: e.clientY };
      render();
    } else if (state.tempLine) {
      // Update temp line end pos
      // We need coords relative to nodesContainer
      state.tempLine.curr = transformMouseToCanvas(e.clientX, e.clientY);
      renderConnections();
    }
  };

  // Mouse Up (End Drag / Connect)
  window.onmouseup = (e) => {
    state.isDraggingNode = null;
    state.isDraggingCanvas = false;

    if (state.tempLine) {
      // Check if dropped on a compatible port
      const target = e.target;
      if (target.classList.contains("port")) {
        const targetNode = target.dataset.node;
        const targetPort = target.dataset.port;
        const targetType = target.dataset.type;

        // Validate connection
        if (
          state.tempLine.startNode !== targetNode &&
          state.tempLine.type !== targetType
        ) {
          // Good connection
          const source =
            state.tempLine.type === "output"
              ? {
                  node: state.tempLine.startNode,
                  port: state.tempLine.startPort,
                }
              : { node: targetNode, port: targetPort };

          const dest =
            state.tempLine.type === "output"
              ? { node: targetNode, port: targetPort }
              : {
                  node: state.tempLine.startNode,
                  port: state.tempLine.startPort,
                };

          // Avoid duplicates
          const exists = state.connections.some(
            (c) =>
              c.sourceNode === source.node &&
              c.sourcePort === source.port &&
              c.targetNode === dest.node &&
              c.targetPort === dest.port,
          );

          if (!exists) {
            state.connections.push({
              sourceNode: source.node,
              sourcePort: source.port,
              targetNode: dest.node,
              targetPort: dest.port,
            });
          }
        }
      }
      state.tempLine = null;
      renderConnections();
    }
  };

  // Background Interaction
  canvasBg.onmousedown = (e) => {
    state.isDraggingCanvas = true;
    state.dragStart = { x: e.clientX, y: e.clientY };
    contextMenu.style.display = "none";

    // Deselect
    state.nodes.forEach((n) => (n.selected = false));
    document
      .querySelectorAll(".node")
      .forEach((n) => n.classList.remove("selected"));
  };

  // Right Click (Context Menu)
  window.oncontextmenu = (e) => {
    // Only on bg
    if (e.target.id === "canvas-bg" || e.target.closest("#canvas-bg")) {
      e.preventDefault();

      // Save pos for creating node (Canvas Coords)
      const pos = transformMouseToCanvas(e.clientX, e.clientY);
      contextMenuPos = pos;

      // Display primarily to measure (visibility hidden trick would be better but block is fine for instant measure)
      contextMenu.style.display = "block";

      // Boundary Logic
      const menuWidth = contextMenu.offsetWidth;
      const menuHeight = contextMenu.offsetHeight;
      const winW = window.innerWidth;
      const winH = window.innerHeight;

      let x = e.clientX;
      let y = e.clientY;

      // X-Axis Flip
      if (x + menuWidth > winW) {
        x -= menuWidth;
      }

      // Y-Axis Flip (Open Upwards if near bottom)
      // If remaining space below is less than menu height (or < 300px safety), flip up
      if (winH - y < 300) {
        contextMenu.style.top = "auto";
        contextMenu.style.bottom = winH - y + "px";
        // Reset origin to bottom-left for animation if we had one
      } else {
        contextMenu.style.bottom = "auto";
        contextMenu.style.top = y + "px";
      }

      contextMenu.style.left = x + "px";
    }
  };

  // Zoom
  window.addEventListener(
    "wheel",
    (e) => {
      // If hovering over context menu, allow default scroll behavior (don't zoom)
      if (e.target.closest("#context-menu")) {
        return;
      }

      if (e.ctrlKey || e.metaKey) {
        // ...
      }
      e.preventDefault();
      const delta = e.deltaY * -0.001;
      state.zoom = Math.min(Math.max(0.5, state.zoom + delta), 2);
      render();
    },
    { passive: false },
  );
}

// Ensure renderConnections logs failures
// Ensure renderConnections logs failures
function renderConnections() {
  if (!svgLayer) {
    console.error("SVG Layer not found!");
    return;
  }
  svgLayer.innerHTML = "";

  // Reset all ports directly via DOM to ensure cleanliness
  // This might be slightly expensive but ensures 100% correct state
  const allPorts = document.querySelectorAll(".port");
  allPorts.forEach((p) => {
    p.style.backgroundColor = "";
    p.style.borderColor = "";
    p.classList.remove("connected");
  });

  // Group connections by Target (for Transform Ordering)
  const targetGroups = new Map();
  state.connections.forEach((conn) => {
    // Only care about transforms for now, but general approach doesn't hurt
    if (conn.targetPort.includes("transform")) {
      const key = `${conn.targetNode}_${conn.targetPort}`;
      if (!targetGroups.has(key)) targetGroups.set(key, []);
      targetGroups.get(key).push(conn);
    }
  });

  // Assign ranks
  const connRanks = new Map(); // connId -> rank
  targetGroups.forEach((conns, key) => {
    if (conns.length > 1) {
      // Sort by source node Y
      conns.sort((a, b) => {
        const nodeA = state.nodes.find((n) => n.id === a.sourceNode);
        const nodeB = state.nodes.find((n) => n.id === b.sourceNode);
        return (nodeA ? nodeA.y : 0) - (nodeB ? nodeB.y : 0);
      });
      conns.forEach((c, idx) => {
        connRanks.set(c.id || JSON.stringify(c), idx + 1); // 1-based index
      });
    }
  });

  state.connections.forEach((conn) => {
    const p1 = getPortPosition(conn.sourceNode, conn.sourcePort, "output");
    const p2 = getPortPosition(conn.targetNode, conn.targetPort, "input");

    if (
      p1 &&
      p2 &&
      Number.isFinite(p1.x) &&
      Number.isFinite(p1.y) &&
      Number.isFinite(p2.x) &&
      Number.isFinite(p2.y)
    ) {
      // Check for rank
      const rank = connRanks.get(conn.id || JSON.stringify(conn));
      drawCurve(p1, p2, false, conn, rank);

      // Colorize Ports
      const sourceNode = state.nodes.find((n) => n.id === conn.sourceNode);
      if (sourceNode) {
        const color = NODE_TYPES[sourceNode.type].color;

        // Colorize Source Port (Output)
        const srcEl = document.querySelector(
          `.port[data-node="${conn.sourceNode}"][data-port="${conn.sourcePort}"][data-type="output"]`,
        );
        if (srcEl) {
          srcEl.style.backgroundColor = color;
          srcEl.style.borderColor = color;
          srcEl.classList.add("connected");
        }

        // Colorize Target Port (Input) - Match the wire flow
        const tgtEl = document.querySelector(
          `.port[data-node="${conn.targetNode}"][data-port="${conn.targetPort}"][data-type="input"]`,
        );
        if (tgtEl) {
          tgtEl.style.backgroundColor = color;
          tgtEl.style.borderColor = color;
          tgtEl.classList.add("connected");
        }
      }
    } else {
      console.warn(
        "Failed to render connection (invalid coords):",
        conn,
        "p1:",
        p1,
        "p2:",
        p2,
      );
    }
  });

  if (state.tempLine) {
    if (state.tempLine.start && state.tempLine.curr) {
      drawCurve(state.tempLine.start, state.tempLine.curr, true);

      // Colorize Start Port during drag
      const startNode = state.nodes.find(
        (n) => n.id === state.tempLine.startNode,
      );
      if (startNode) {
        const color = NODE_TYPES[startNode.type].color;
        const type = state.tempLine.type === "output" ? "output" : "input"; // Origin is always startPort
        const portEl = document.querySelector(
          `.port[data-node="${state.tempLine.startNode}"][data-port="${state.tempLine.startPort}"][data-type="${type}"]`,
        );
        if (portEl) {
          portEl.style.backgroundColor = color;
          portEl.style.borderColor = color;
        }
      }
    } else {
      console.warn("Invalid tempLine coords", state.tempLine);
    }
  }
}
