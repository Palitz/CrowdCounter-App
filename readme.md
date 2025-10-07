\# High-Performance Real-Time Crowd Counting



This project provides a lightweight and efficient application for real-time crowd counting and heatmap generation from a video source. It leverages a state-of-the-art pre-trained model (DM-Count) within a high-performance inference engine, making it suitable for deployment on edge devices like the NVIDIA Jetson and Raspberry Pi.



!\[Heatmap Demo](https://i.imgur.com/your-demo-image.gif)  <!-- Optional: You can create a GIF of your app running and upload it to a site like Imgur -->



---



\## Features



\- \*\*High Accuracy:\*\* Utilizes the DM-Count model, which excels at accurate counting and avoids common false positives.

\- \*\*Real-Time Performance:\*\* Optimized, in-memory inference loop provides a high FPS suitable for live video.

\- \*\*Heatmap Generation:\*\* Generates intuitive heatmaps to visualize crowd density.

\- \*\*Cross-Platform:\*\* Includes scripts and instructions for exporting the model to ONNX for deployment on both ARM (Raspberry Pi) and CUDA-enabled (Jetson) devices.

\- \*\*Self-Contained:\*\* The application is robust and relies on a local model cache, ensuring it will continue to work even if external libraries change.



---



\## Getting Started



These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.



\### Prerequisites



\- Python 3.8+

\- An NVIDIA GPU is recommended for best performance on a PC, but it will run on a CPU.



\### Installation



1\.  \*\*Clone the repository:\*\*

&nbsp;   ```bash

&nbsp;   git clone https://github.com/your-username/CrowdCounter-App.git

&nbsp;   cd CrowdCounter-App

&nbsp;   ```



2\.  \*\*Install the required Python packages:\*\*

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



---



\## Usage



The project is split into three main steps: running the main application, exporting the model, and deploying on an edge device.



\### 1. Running the Main Application (PC)



The `app.py` script is the main application for testing on your development machine.



\- \*\*Configuration:\*\* Open `app.py` and set the `VIDEO\_PATH` variable to your test video file. You can also adjust the `SCALE\_PERCENT`.



\- \*\*Run the app:\*\*

&nbsp;   ```bash

&nbsp;   python app.py

&nbsp;   ```

&nbsp;   The first time you run this, it will download the pre-trained DM-Count model (approx. 86 MB) to a local cache. A window will then appear showing the video with the real-time count, FPS, and heatmap overlay.



\### 2. Exporting the Model for Deployment



To prepare the model for edge devices, you need to convert it to the ONNX format.



\- \*\*Run the export script:\*\*

&nbsp;   ```bash

&nbsp;   python export\_onnx.py

&nbsp;   ```

&nbsp;   This will create a file named `dm\_count.onnx` in the project directory. This is the file you will move to your edge device.



\### 3. Deploying on an Edge Device



Copy the `dm\_count.onnx` file to your Raspberry Pi or Jetson.



\#### On a Raspberry Pi:



1\.  \*\*Install dependencies:\*\*

&nbsp;   ```bash

&nbsp;   pip install opencv-python-headless numpy onnxruntime

&nbsp;   ```

2\.  \*\*Run the inference script:\*\*

&nbsp;   - Open `rpi\_inference.py` and set the `VIDEO\_PATH`.

&nbsp;   - Execute the script: `python rpi\_inference.py`



\#### On an NVIDIA Jetson:



1\.  \*\*Install dependencies (if needed):\*\*

&nbsp;   ```bash

&nbsp;   pip install numpy onnxruntime-gpu

&nbsp;   ```

&nbsp;   \*Note: Using `onnxruntime-gpu` will allow it to leverage the TensorRT execution provider for maximum speed.\*

2\.  \*\*Run the inference script:\*\*

&nbsp;   - Open `jetson\_inference.py` and set the `VIDEO\_PATH`.

&nbsp;   - Execute the script: `python jetson\_inference.py`



---



\## Acknowledgements



\- This project utilizes the DM-Count model, made available through the `lwcc` library.

\- Our journey through model selection, custom training, and debugging led to this final, robust implementation.

