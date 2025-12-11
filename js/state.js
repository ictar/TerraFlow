/* Global State */
const state = {
    nodes: [],
    connections: [],
    pan: { x: 0, y: 0 },
    zoom: 1,
    isDraggingNode: null,
    isDraggingCanvas: false,
    dragStart: { x: 0, y: 0 },
    tempLine: null 
};

// Global DOM References (populated on load)
let canvasBg;
let nodesContainer;
let svgLayer;
let contextMenu;
let contextMenuPos = { x: 0, y: 0 };
