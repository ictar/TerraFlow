/* Initialization */

document.addEventListener('DOMContentLoaded', () => {
    // Populate Bindings
    canvasBg = document.getElementById('canvas-bg');
    nodesContainer = document.getElementById('nodes-container');
    svgLayer = document.getElementById('connections-layer');
    contextMenu = document.getElementById('context-menu');
    
    // Init Events
    initUIEvents();
    
    // Create Initial Graph (Wait a bit for fonts/styles)
    setTimeout(() => {
        // --- Nodes ---
        const modelNode = addNode('ModelFactory', 100, 150);
        const taskNode = addNode('SegmentationTask', 500, 220);
        const optimNode = addNode('OptimizerConfig', 100, 380);
        const dataNode = addNode('HLSDataModule', 500, 520);
        const trainerNode = addNode('TrainerConfig', 900, 380);

        // --- Connections ---
        state.connections.push(
            { sourceNode: modelNode.id, sourcePort: 'model_args', targetNode: taskNode.id, targetPort: 'model_args' },
            { sourceNode: optimNode.id, sourcePort: 'optim_args', targetNode: taskNode.id, targetPort: 'optim_args' },
            { sourceNode: taskNode.id, sourcePort: 'task', targetNode: trainerNode.id, targetPort: 'task' },
            { sourceNode: dataNode.id, sourcePort: 'datamodule', targetNode: trainerNode.id, targetPort: 'datamodule' }
        );

        renderConnections();
    }, 100);
});
