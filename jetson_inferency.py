# jetson_inference.py

import cv2
import numpy as np
import onnxruntime as rt
import time
import matplotlib.pyplot as plt

# --- Configuration ---
ONNX_MODEL_PATH = "dm_count.onnx"
# IMPORTANT: Set this to your video path or a GStreamer pipeline for a CSI camera on the Jetson
VIDEO_PATH = "path/to/your/video.mp4" 
# The Jetson can handle a larger size, so we can use 50% or more
SCALE_PERCENT = 50 

# --- Initialize ONNX Runtime with TensorRT ---
print("--- Initializing ONNX Runtime for TensorRT... ---")
# The key difference: we prioritize TensorRT and CUDA providers for GPU acceleration
sess = rt.InferenceSession(ONNX_MODEL_PATH, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
print("--- ONNX Runtime initialized. ---")

# --- Video Processing Loop ---
cap = cv2.VideoCapture(VIDEO_PATH)
prev_frame_time = 0

print("\n--- Starting Jetson inference... Press 'q' to quit. ---")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- Resize and Pre-process ---
    width = int(frame.shape[1] * SCALE_PERCENT / 100)
    height = int(frame.shape[0] * SCALE_PERCENT / 100)
    frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Manually normalize and transpose
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    input_tensor = np.transpose(frame_rgb, (2, 0, 1)) # HWC to CHW
    input_tensor = (input_tensor / 255.0 - mean) / std
    input_tensor = input_tensor[np.newaxis, :, :, :].astype('float32') # Add batch dimension

    # --- Run Inference ---
    results = sess.run(None, {input_name: input_tensor})
    density_map = results[0]
    count = np.sum(density_map)
    
    # --- Display Info (Jetson can handle the heatmap) ---
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time
    
    # Generate and overlay heatmap
    density_map_squeezed = density_map.squeeze()
    norm_density_map = (density_map_squeezed - density_map_squeezed.min()) / (density_map_squeezed.max() - density_map_squeezed.min() + 1e-6)
    heatmap = plt.get_cmap('jet')(norm_density_map)[:, :, :3]
    heatmap_bgr = cv2.cvtColor((heatmap * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap_bgr, (width, height))
    overlay = cv2.addWeighted(frame_resized, 0.6, heatmap_resized, 0.4, 0)
    
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(overlay, f"Count: {int(count)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Jetson Inference", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()