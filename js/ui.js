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

function drawCurve(p1, p2, isTemp, connData = null, orderIndex = null) {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
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
    
    path.setAttribute('d', d);
    path.setAttribute('class', isTemp ? 'drag-line' : 'connection');
    
    // Colorize based on Source Node
    let sourceNodeId = null;
    if (isTemp && state.tempLine) {
        sourceNodeId = state.tempLine.startNode;
    } else if (connData) {
        sourceNodeId = connData.sourceNode;
    }
    
    let strokeColor = null;
    if (sourceNodeId) {
        const node = state.nodes.find(n => n.id === sourceNodeId);
        if (node && NODE_TYPES[node.type]) {
            strokeColor = NODE_TYPES[node.type].color;
            path.style.stroke = strokeColor;
        }
    }
    
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

        const midX = mt3 * p1.x + 3 * mt2 * t * cp1.x + 3 * mt * t2 * cp2.x + t3 * p2.x;
        const midY = mt3 * p1.y + 3 * mt2 * t * cp1.y + 3 * mt * t2 * cp2.y + t3 * p2.y;

        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('transform', `translate(${midX}, ${midY})`);
        
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('r', '10');
        circle.setAttribute('fill', strokeColor || '#555');
        circle.setAttribute('stroke', '#121212');
        circle.setAttribute('stroke-width', '2');
        
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('dy', '3'); // Vertical center adjustment
        text.setAttribute('fill', '#fff');
        text.setAttribute('font-size', '10px');
        text.setAttribute('font-weight', 'bold');
        text.textContent = orderIndex;

        g.appendChild(circle);
        g.appendChild(text);
        g.style.pointerEvents = 'none'; // Click through to wire
        
        svgLayer.appendChild(g);
    }
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
    const START_X = 50;
    const START_Y = 100;
    const COL_WIDTH = 320; // Tighter column width
    const NODE_GAP = 40;   // Vertical gap between nodes

    // Sort keys to process left-to-right
    const sortedLevels = Array.from(levelGroups.keys()).sort((a,b) => a - b);
    
    sortedLevels.forEach(lvl => {
        const nodeIds = levelGroups.get(lvl);
        
        // Sort nodes in layer by type to keep graph organized
        nodeIds.sort((a, b) => {
             const nA = state.nodes.find(n => n.id === a);
             const nB = state.nodes.find(n => n.id === b);
             return nA.type.localeCompare(nB.type);
        });

        let currentY = START_Y;

        nodeIds.forEach(id => {
            const node = state.nodes.find(n => n.id === id);
            if (!node) return;

            // Set Position
            node.x = START_X + lvl * COL_WIDTH;
            node.y = currentY;

            // DOM Update
            const el = document.getElementById(node.id);
            if (el) {
                el.style.left = node.x + 'px';
                el.style.top = node.y + 'px';
            }

            // Estimate Node Height for next position
            // Base: ~50px (header) + Padding
            // Params: ~50px each (label + input)
            // Ports: ~24px each
            const typeDef = NODE_TYPES[node.type] || {};
            const paramCount = (typeDef.params || []).length;
            const inputCount = (typeDef.inputs || []).length;
            const outputCount = (typeDef.outputs || []).length;
            const socketRows = Math.max(inputCount, outputCount) + (inputCount > 0 && outputCount > 0 ? 0 : 0); // Rough est
            
            // Refined estimation
            // Header: 40
            // Params: 60 * paramCount
            // Sockets: 30 * (input + output) -- actually they are stacked unless specific logic
            // ui.js addNode logic: inputs, then outputs, then params.
            // Let's assume sequential stacking for safety.
            
            const estimatedHeight = 60 + (paramCount * 60) + (inputCount * 30) + (outputCount * 30);
            
            currentY += estimatedHeight + NODE_GAP;
        });
    });

    renderConnections();
    state.zoom = 1;
    render();
}

// --- Global Settings Logic ---

function updateGlobalSetting(key, value) {
    if (key === 'seed_everything') {
        state.globalConfig.seed_everything = parseInt(value) || 0;
    }
}

function updateGlobalUI() {
    const seedEl = document.getElementById('global-seed');
    if (seedEl) {
        seedEl.value = state.globalConfig.seed_everything;
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
    
    // Reset all ports directly via DOM to ensure cleanliness
    // This might be slightly expensive but ensures 100% correct state
    const allPorts = document.querySelectorAll('.port');
    allPorts.forEach(p => {
        p.style.backgroundColor = '';
        p.style.borderColor = '';
        p.classList.remove('connected');
    });

    // Group connections by Target (for Transform Ordering)
    const targetGroups = new Map();
    state.connections.forEach(conn => {
        // Only care about transforms for now, but general approach doesn't hurt
        if (conn.targetPort.includes('transform')) {
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
                const nodeA = state.nodes.find(n => n.id === a.sourceNode);
                const nodeB = state.nodes.find(n => n.id === b.sourceNode);
                return (nodeA ? nodeA.y : 0) - (nodeB ? nodeB.y : 0);
            });
            conns.forEach((c, idx) => {
                connRanks.set(c.id || JSON.stringify(c), idx + 1); // 1-based index
            });
        }
    });

    state.connections.forEach(conn => {
        const p1 = getPortPosition(conn.sourceNode, conn.sourcePort, 'output');
        const p2 = getPortPosition(conn.targetNode, conn.targetPort, 'input');
        
        if (p1 && p2 && Number.isFinite(p1.x) && Number.isFinite(p1.y) && Number.isFinite(p2.x) && Number.isFinite(p2.y)) {
            // Check for rank
            const rank = connRanks.get(conn.id || JSON.stringify(conn));
            drawCurve(p1, p2, false, conn, rank);
            
            // Colorize Ports
            const sourceNode = state.nodes.find(n => n.id === conn.sourceNode);
            if (sourceNode) {
                const color = NODE_TYPES[sourceNode.type].color;
                
                // Colorize Source Port (Output)
                const srcEl = document.querySelector(`.port[data-node="${conn.sourceNode}"][data-port="${conn.sourcePort}"][data-type="output"]`);
                if (srcEl) {
                    srcEl.style.backgroundColor = color;
                    srcEl.style.borderColor = color;
                    srcEl.classList.add('connected');
                }
                
                // Colorize Target Port (Input) - Match the wire flow
                const tgtEl = document.querySelector(`.port[data-node="${conn.targetNode}"][data-port="${conn.targetPort}"][data-type="input"]`);
                if (tgtEl) {
                    tgtEl.style.backgroundColor = color;
                    tgtEl.style.borderColor = color;
                    tgtEl.classList.add('connected');
                }
            }

        } else {
            console.warn('Failed to render connection (invalid coords):', conn, 'p1:', p1, 'p2:', p2);
        }
    });

    if (state.tempLine) {
        if (state.tempLine.start && state.tempLine.curr) {
             drawCurve(state.tempLine.start, state.tempLine.curr, true);
             
             // Colorize Start Port during drag
             const startNode = state.nodes.find(n => n.id === state.tempLine.startNode);
             if (startNode) {
                 const color = NODE_TYPES[startNode.type].color;
                 const type = state.tempLine.type === 'output' ? 'output' : 'input'; // Origin is always startPort
                 const portEl = document.querySelector(`.port[data-node="${state.tempLine.startNode}"][data-port="${state.tempLine.startPort}"][data-type="${type}"]`);
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
