import torch
from lwcc import LWCC

print("--- Loading the pre-trained DM-Count model from LWCC cache... ---")
# 1. Load the model object just like in our app
model = LWCC.load_model('DM-Count', 'SHB')
model.eval() # Set to evaluation mode

# 2. Define a dummy input tensor
# The size must match what the model expects during inference.
# We'll use the resized dimensions from our app.
# Let's assume a 1920x1080 video scaled by 50% -> 960x540
# Batch size = 1, Channels = 3 (RGB), Height = 540, Width = 960
dummy_input = torch.randn(1, 3, 540, 960) 

# 3. Define the output filename
onnx_file_path = "dm_count.onnx"

print(f"--- Exporting model to ONNX format at '{onnx_file_path}'... ---")
# 4. Export the model
torch.onnx.export(
    model,                  # The model to export
    dummy_input,            # A sample input tensor
    onnx_file_path,         # Where to save the model
    export_params=True,     # Store the trained weights in the model file
    opset_version=11,       # A standard ONNX version
    do_constant_folding=True, # Execute constant folding for optimization
    input_names = ['input'],   # Name for the input layer in the ONNX model
    output_names = ['output'], # Name for the output layer in the ONNX model
)

print("\n--- ONNX Export Complete! ---")
print(f"Your model has been saved to '{onnx_file_path}'.")
print("This file is now ready for deployment on a Jetson (with TensorRT) or Raspberry Pi (with ONNX Runtime).")
