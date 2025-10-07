# app.py

import torch
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from lwcc import LWCC

class CrowdCounter:
    """
    A robust class to handle crowd counting and heatmap generation.
    It encapsulates model loading and the entire inference pipeline.
    """
    def __init__(self, model_name='DM-Count', model_weights='SHB'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Initializing CrowdCounter on device: {self.device} ---")
        
        # Load the model once during initialization
        self.model = LWCC.load_model(model_name, model_weights)
        self.model.to(self.device)
        self.model.eval()
        
        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("--- Model loaded successfully ---")

    def process_frame(self, frame):
        """
        Processes a single video frame to get the crowd count and an overlayed heatmap.
        
        Args:
            frame (np.ndarray): The video frame (in BGR format from OpenCV).
            
        Returns:
            tuple: A tuple containing:
                - overlay (np.ndarray): The original frame with the heatmap and info.
                - count (float): The estimated crowd count.
        """
        # 1. Prepare Frame for Model (Manually)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

        # 2. Run Inference
        with torch.no_grad():
            pred_density_map = self.model(img_tensor)
        
        count = torch.sum(pred_density_map).item()
        
        # 3. Generate Heatmap for Visualization
        density_map_np = pred_density_map.squeeze().cpu().numpy()
        norm_density_map = (density_map_np - density_map_np.min()) / (density_map_np.max() - density_map_np.min() + 1e-6)
        heatmap = plt.get_cmap('jet')(norm_density_map)[:, :, :3]
        heatmap_bgr = cv2.cvtColor((heatmap * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Resize heatmap and overlay it
        heatmap_resized = cv2.resize(heatmap_bgr, (frame.shape[1], frame.shape[0]))
        overlay = cv2.addWeighted(frame, 0.6, heatmap_resized, 0.4, 0)
        
        return overlay, count

# ==============================================================================
# ---  Main Execution Block ---
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    VIDEO_PATH = 'C:/Users/John Thomas/Downloads/4323179-hd_1920_1080_30fps.mp4'
    SCALE_PERCENT = 50

    # 1. Create an instance of our CrowdCounter
    crowd_counter = CrowdCounter(model_name='DM-Count', model_weights='SHB')

    # 2. Set up video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    prev_frame_time = 0
    
    print("\n--- Starting video processing... Press 'q' to quit. ---")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        width = int(frame.shape[1] * SCALE_PERCENT / 100)
        height = int(frame.shape[0] * SCALE_PERCENT / 100)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # 3. Process the frame using our class method
        overlay_frame, count = crowd_counter.process_frame(frame)

        # 4. Display FPS and Count
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        
        cv2.putText(overlay_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay_frame, f"Count: {int(count)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Crowd Counter Implementation", overlay_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n--- Video processing finished. ---")
    cap.release()
    cv2.destroyAllWindows()