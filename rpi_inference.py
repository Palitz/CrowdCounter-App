# rpi_inference.py

import cv2
import numpy as np
import onnxruntime as rt
import time

# --- Configuration ---
ONNX_MODEL_PATH = "dm_count.onnx"
# IMPORTANT: Set this to the path of a video on your Raspberry Pi, or '0' for a PiCamera
VIDEO_PATH = "path/to/your/video.mp4" 
# Use a smaller size for better FPS on the Pi. 25-30% is a good starting point.
SCALE_PERCENT = 30 

# --- Initialize ONNX Runtime for CPU ---
print("--- Initializing ONNX Runtime for CPU... ---")
# Create an inference session, specifying the CPUExecutionProvider for clarity
sess = rt.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
print("--- ONNX Runtime initialized. ---")

# --- Video Processing Loop ---
cap = cv2.VideoCapture(VIDEO_PATH)
prev_frame_time = 0

print("\n--- Starting Raspberry Pi inference... Press 'q' to quit. ---")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- Resize and Pre-process ---
    width = int(frame.shape[1] * SCALE_PERCENT / 100)
    height = int(frame.shape[0] * SCALE_PERCENT / 100)
    frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Manually normalize and transpose the image data
    # Normalization: [0, 255] -> [0, 1] and then standardize
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    input_tensor = np.transpose(frame_rgb, (2, 0, 1)) # HWC to CHW
    input_tensor = (input_tensor / 255.0 - mean) / std
    input_tensor = input_tensor[np.newaxis, :, :, :].astype('float32') # Add batch dimension

    # --- Run Inference ---
    results = sess.run(None, {input_name: input_tensor})
    density_map = results[0]
    count = np.sum(density_map)
    
    # --- Display Info (Heatmap is skipped for performance) ---
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time
    
    # Draw info on the original, larger frame for better visibility
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Count: {int(count)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # To display the video, you need a desktop environment on the Pi.
    # If running headless (e.g., via SSH), you would comment out this section.
    cv2.imshow("Raspberry Pi Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()