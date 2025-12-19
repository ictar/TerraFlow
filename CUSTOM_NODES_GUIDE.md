Custom Nodes Implementation Guide:

1. **Custom Nodes Added**:
   - Every category (Data, Architecture, Task, Optimizer, Scheduler, Logger, Callback) now has a "Custom" counterpart (e.g., `CustomDataModule`, `CustomBackbone`, `CustomTask`).
   - These nodes appear in the context menu (Right Click > Category).

2. **Features**:
   - **Class Path**: Each Custom Node allows you to specify a custom python class path (e.g., `my_module.MyModel` or `torch.optim.Adam`).
   - **Arbitrary Parameters**:
     - You can add any number of parameters to these nodes.
     - Click "Add Param" in the node UI.
     - Provide a "Key" (argument name) and "Value".
     - Values are automatically parsed (Numbers are converted to int/float, 'True'/'False' to booleans, otherwise strings).

3. **Export**:
   - The YAML/Notebook export logic has been updated to include these custom parameters in the `init_args` or relevant configuration sections.
   - This allows you to integrate any external or custom code into the TerraFlow graph seamlessly.
