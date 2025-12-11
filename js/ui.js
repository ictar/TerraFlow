/* UI Logic: Rendering, Interactions, and Node Management */

function render() {
    nodesContainer.style.transform = `scale(${state.zoom}) translate(${state.pan.x}px, ${state.pan.y}px)`;
    // svgLayer is a child of nodesContainer, so it inherits the transform automatically. 
    // Applying it again causes double-scaling/translation.
    renderConnections();
}

// --- DOM Creation ---

function createNodeDOM(node) {
    const el = document.createElement('div');
    el.className = `node ${node.selected ? 'selected' : ''}`;
    el.id = node.id;
    el.style.left = node.x + 'px';
    el.style.top = node.y + 'px';
    el.style.borderColor = NODE_TYPES[node.type].color;
    
    // Header
    const header = document.createElement('div');
    header.className = 'node-header';
    header.style.background = NODE_TYPES[node.type].color + '44'; // Low opacity bg
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
        state.nodes.forEach(n => n.selected = false);
        node.selected = true;
        document.querySelectorAll('.node').forEach(n => n.classList.remove('selected'));
        el.classList.add('selected');
    };
    el.appendChild(header);

    // Body
    const body = document.createElement('div');
    body.className = 'node-body';

    // Inputs
    NODE_TYPES[node.type].inputs.forEach(inputKey => {
        const row = document.createElement('div');
        row.className = 'socket-row input';
        row.innerHTML = `<span class="socket-label">${formatPortName(inputKey)}</span>`;
        
        const socket = document.createElement('div');
        socket.className = 'port input';
        socket.dataset.port = inputKey;
        socket.dataset.node = node.id;
        socket.dataset.type = 'input';
        
        // Wire connect logic
        socket.onmousedown = (e) => startLineDrag(e, node.id, inputKey, 'input');

        row.prepend(socket);
        body.appendChild(row);
    });

    // Params
    NODE_TYPES[node.type].params.forEach(param => {
        const wrapper = document.createElement('div');
        wrapper.className = 'node-input-wrapper';
        wrapper.innerHTML = `<span class="node-input-label">${param.label}</span>`;
        
        let input;
        if (param.type === 'select') {
            input = document.createElement('select');
            input.className = 'node-input';
            input.id = `${node.id}_${param.key}`;
            input.name = `${node.id}_${param.key}`;
            input.setAttribute('autocomplete', 'off'); // Prevent autofill suggestions
            param.options.forEach(opt => {
                const optEl = document.createElement('option');
                optEl.value = opt;
                optEl.innerText = opt;
                if(opt === node.data[param.key]) optEl.selected = true;
                input.appendChild(optEl);
            });
        } else {
            input = document.createElement('input');
            input.className = 'node-input';
            input.type = param.type === 'number' ? 'number' : 'text';
            input.value = node.data[param.key];
            input.id = `${node.id}_${param.key}`;
            input.name = `${node.id}_${param.key}`;
            input.setAttribute('autocomplete', 'off'); // Prevent autofill suggestions
        }
        
        input.oninput = (e) => { node.data[param.key] = e.target.value; };
        // Prevent pan when typing
        input.onmousedown = (e) => e.stopPropagation(); 
        
        wrapper.appendChild(input);
        body.appendChild(wrapper);
    });

    // Outputs
    NODE_TYPES[node.type].outputs.forEach(outputKey => {
        const row = document.createElement('div');
        row.className = 'socket-row output';
        row.innerHTML = `<span class="socket-label">${formatPortName(outputKey)}</span>`;
        
        const socket = document.createElement('div');
        socket.className = 'port output';
        socket.dataset.port = outputKey;
        socket.dataset.node = node.id;
        socket.dataset.type = 'output';

        // Wire connect logic
        socket.onmousedown = (e) => startLineDrag(e, node.id, outputKey, 'output');

        row.appendChild(socket);
        body.appendChild(row);
    });

    el.appendChild(body);
    nodesContainer.appendChild(el);
}

function formatPortName(key) {
    return key.replace(/_/g, ' ').toUpperCase();
}

// --- Connections ---

// function renderConnections() { ... } // Removed duplicate definition here. Using the one defined at the bottom.

function drawCurve(p1, p2, isTemp, connData = null) {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    // Dynamic Curvature:
    // If nodes are far apart horizontally, start with a wide curve.
    // If close or vertical, ensure enough "bulge" to look like a wire.
    const deltaX = Math.abs(p2.x - p1.x);
    // Control point distance: at least 60px, or 50% of the horizontal distance
    const dist = Math.max(deltaX * 0.5, 60); 
    
    // Wire flow: always out to the right (p1 + dist) and in from the left (p2 - dist)
    const cp1 = { x: p1.x + dist, y: p1.y };
    const cp2 = { x: p2.x - dist, y: p2.y };
    
    // Improve vertical handling: If going backwards (p2.x < p1.x), loop around
    if (p2.x < p1.x + 20) {
        // Create a C-shape loop
        cp1.x = Math.max(p1.x + 200, p1.x + deltaX);
        cp2.x = Math.min(p2.x - 200, p2.x - deltaX);
        // Push control points vertically apart to avoid cutting through nodes? 
        // Simple cubic bezier is usually fine if CP range is wide enough.
    }
    
    const d = `M ${p1.x} ${p1.y} C ${cp1.x} ${cp1.y} ${cp2.x} ${cp2.y} ${p2.x} ${p2.y}`;
    
    path.setAttribute('d', d);
    path.setAttribute('class', isTemp ? 'drag-line' : 'connection');
    
    if (!isTemp && connData) {
        path.oncontextmenu = (e) => {
            e.preventDefault();
            e.stopPropagation();
            if(confirm("Delete this connection?")) {
                deleteConnection(connData);
            }
        };
    }
    
    svgLayer.appendChild(path);
}

function getPortPosition(nodeId, portName, type) {
    const nodeEl = document.getElementById(nodeId);
    if (!nodeEl) return null;
    const portEl = nodeEl.querySelector(`.port[data-port="${portName}"][data-type="${type}"]`);
    if (!portEl) return null;
    
    // Reliable approach: Use getBoundingClientRect delta, scaled by zoom
    const portRect = portEl.getBoundingClientRect();
    const containerRect = nodesContainer.getBoundingClientRect();

    // Calculate relative position within the zoomed/panned container context
    const x = (portRect.left - containerRect.left) / state.zoom + (portRect.width / 2) / state.zoom;
    const y = (portRect.top - containerRect.top) / state.zoom + (portRect.height / 2) / state.zoom;
    
    return { x, y };
}

// --- Coordinates Helper ---

function transformMouseToCanvas(clientX, clientY) {
    const rect = nodesContainer.getBoundingClientRect();
    return {
        x: (clientX - rect.left) / state.zoom,
        y: (clientY - rect.top) / state.zoom
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
        curr: startPos
    };
}

function deleteNode(id) {
    if(!confirm("Delete node?")) return;
    state.nodes = state.nodes.filter(n => n.id !== id);
    state.connections = state.connections.filter(c => c.sourceNode !== id && c.targetNode !== id);
    
    // DOM cleanup
    const el = document.getElementById(id);
    if(el) el.remove();
    
    renderConnections();
}

function deleteConnection(conn) {
    state.connections = state.connections.filter(c => c !== conn);
    renderConnections();
}

// --- Global Actions for HTML Buttons ---

function addNode(type, x, y, initialData = {}) {
    // Use a random suffix to ensure uniqueness even if called rapidly
    const id = 'node_' + Date.now() + '_' + Math.floor(Math.random() * 1000);
    const newNode = {
        id, type, x: x || 100, y: y || 100, data: { ...initialData }
    };
    
    // Fill missing params with defaults
    NODE_TYPES[type].params.forEach(p => {
        if (newNode.data[p.key] === undefined) newNode.data[p.key] = p.default;
    });

    state.nodes.push(newNode);
    createNodeDOM(newNode);
    contextMenu.style.display = 'none';
    return newNode;
}

function clearCanvas() {
    if(!confirm("Clear all?")) return;
    state.nodes = [];
    state.connections = [];
    nodesContainer.innerHTML = '';
    nodesContainer.appendChild(svgLayer); // Restore SVG
    svgLayer.innerHTML = '';
}

function resetView() {
    state.pan = { x: 0, y: 0 };
    state.zoom = 1;
    render();
}

function autoLayout() {
    // 1. Build Graph & In-Degrees
    const graph = new Map();
    const inDegree = new Map();
    state.nodes.forEach(node => {
         graph.set(node.id, []);
         inDegree.set(node.id, 0);
    });

    state.connections.forEach(conn => {
        // Directed edge: source -> target
        const adj = graph.get(conn.sourceNode) || [];
        adj.push(conn.targetNode);
        graph.set(conn.sourceNode, adj);
        
        inDegree.set(conn.targetNode, (inDegree.get(conn.targetNode) || 0) + 1);
    });

    // 2. Assign Levels (Longest path from source)
    // Initialize levels
    const levels = new Map();
    state.nodes.forEach(n => levels.set(n.id, 0));

    // Simple robust leveling: iterate |N| times relax edges
    // This handles any acyclic graph correctly without complex topological sort impl
    for (let i = 0; i < state.nodes.length + 2; i++) {
        state.connections.forEach(conn => {
            const srcLvl = levels.get(conn.sourceNode);
            const tgtLvl = levels.get(conn.targetNode);
            if (srcLvl + 1 > tgtLvl) {
                levels.set(conn.targetNode, srcLvl + 1);
            }
        });
    }

    // 3. Group Nodes by Level
    const levelGroups = new Map();
    levels.forEach((lvl, id) => {
        if (!levelGroups.has(lvl)) levelGroups.set(lvl, []);
        levelGroups.get(lvl).push(id);
    });

    // 4. Assign Grid Positions
    const START_X = 100;
    const START_Y = 150;
    const COL_WIDTH = 380; // Wide enough for nodes
    const ROW_HEIGHT = 400; // Tall enough for large nodes

    // Sort keys to process left-to-right
    const sortedLevels = Array.from(levelGroups.keys()).sort((a,b) => a - b);
    
    // Track vertical cursor per layer? No, just stack them.
    // To vertically center, get max height of a column?
    // Let's keep it simple: Top alignment for now.
    
    sortedLevels.forEach(lvl => {
        const nodeIds = levelGroups.get(lvl);
        
        // Optional: Sort nodes in layer by their Y position or type to keep graph stable?
        // Or just alphabetical/type to look consistent
        nodeIds.sort((a, b) => {
             const nA = state.nodes.find(n => n.id === a);
             const nB = state.nodes.find(n => n.id === b);
             // Heuristic: Input types on top? 
             return nA.type.localeCompare(nB.type);
        });

        nodeIds.forEach((id, idx) => {
            const node = state.nodes.find(n => n.id === id);
            if (!node) return;

            // Animate? No, just set.
            node.x = START_X + lvl * COL_WIDTH;
            node.y = START_Y + idx * ROW_HEIGHT;

            // DOM Update
            const el = document.getElementById(node.id);
            if (el) {
                el.style.left = node.x + 'px';
                el.style.top = node.y + 'px';
            }
        });
    });

    renderConnections();
    // Reset view to center content (optional)?
    // Let's just fit view loosely? No, keep user zoom.
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
            el.style.left = state.isDraggingNode.x + 'px';
            el.style.top = state.isDraggingNode.y + 'px';
            
            state.dragStart = { x: e.clientX, y: e.clientY };
            renderConnections();
        } 
        else if (state.isDraggingCanvas) {
            state.pan.x += e.clientX - state.dragStart.x;
            state.pan.y += e.clientY - state.dragStart.y;
            state.dragStart = { x: e.clientX, y: e.clientY };
            render();
        }
        else if (state.tempLine) {
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
            if (target.classList.contains('port')) {
                const targetNode = target.dataset.node;
                const targetPort = target.dataset.port;
                const targetType = target.dataset.type;
                
                // Validate connection
                if (state.tempLine.startNode !== targetNode && state.tempLine.type !== targetType) {
                    // Good connection
                    const source = state.tempLine.type === 'output' ? 
                        { node: state.tempLine.startNode, port: state.tempLine.startPort } :
                        { node: targetNode, port: targetPort };
                        
                    const dest = state.tempLine.type === 'output' ?
                        { node: targetNode, port: targetPort } :
                        { node: state.tempLine.startNode, port: state.tempLine.startPort };
                        
                    // Avoid duplicates
                    const exists = state.connections.some(c => 
                        c.sourceNode === source.node && c.sourcePort === source.port &&
                        c.targetNode === dest.node && c.targetPort === dest.port
                    );
                    
                    if (!exists) {
                        state.connections.push({
                            sourceNode: source.node, sourcePort: source.port,
                            targetNode: dest.node, targetPort: dest.port
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
        contextMenu.style.display = 'none';
        
        // Deselect
        state.nodes.forEach(n => n.selected = false);
        document.querySelectorAll('.node').forEach(n => n.classList.remove('selected'));
    };

    // Right Click (Context Menu)
    window.oncontextmenu = (e) => {
        // Only on bg
        if (e.target.id === 'canvas-bg' || e.target.closest('#canvas-bg')) {
            e.preventDefault();
            contextMenu.style.display = 'block';
            contextMenu.style.left = e.clientX + 'px';
            contextMenu.style.top = e.clientY + 'px';
            
            // Save pos for creating node
            const pos = transformMouseToCanvas(e.clientX, e.clientY);
            contextMenuPos = pos;
        }
    };

    // Zoom
    window.addEventListener('wheel', (e) => {
        if (e.ctrlKey || e.metaKey) { // Only zoom with Ctrl/Meta key to avoid interfering with normal scroll? Or just always zoom as per original?
             // Original behavior was always zoom
        }
        e.preventDefault();
        const delta = e.deltaY * -0.001;
        state.zoom = Math.min(Math.max(0.5, state.zoom + delta), 2);
        render();
    }, { passive: false });
}

// Ensure renderConnections logs failures
// Ensure renderConnections logs failures
function renderConnections() {
    if (!svgLayer) {
        console.error("SVG Layer not found!");
        return;
    }
    svgLayer.innerHTML = '';
    
    state.connections.forEach(conn => {
        const p1 = getPortPosition(conn.sourceNode, conn.sourcePort, 'output');
        const p2 = getPortPosition(conn.targetNode, conn.targetPort, 'input');
        
        if (p1 && p2 && Number.isFinite(p1.x) && Number.isFinite(p1.y) && Number.isFinite(p2.x) && Number.isFinite(p2.y)) {
            drawCurve(p1, p2, false, conn);
        } else {
            console.warn('Failed to render connection (invalid coords):', conn, 'p1:', p1, 'p2:', p2);
        }
    });

    if (state.tempLine) {
        if (state.tempLine.start && state.tempLine.curr) {
             drawCurve(state.tempLine.start, state.tempLine.curr, true);
        } else {
             console.warn("Invalid tempLine coords", state.tempLine);
        }
    }
}
