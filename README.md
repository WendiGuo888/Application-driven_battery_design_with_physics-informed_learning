Data and code necessary to replicate the results in the paper "Bridging battery design and health assessment through virtual sensing and physics-informed learning".


The battery aging dataset is publicly available at https://doi.org/10.5281/ zenodo.16538328 and https://doi.org/10.5281/zenodo.20679088.
If you make use of this dataset, please cite this paper.

I - System requirements
-----
All the analyses conducted to generate the results have been conducted in Python. 
Python (version 3.8 or higher). All necessary python packages required to run the code are specified in requirements.txt. All scripts have been tested on Windows 11.
No non-standard hardware is required. (Note: A CUDA-enabled NVIDIA GPU is recommended for accelerating the PINN training but not strictly required; the code runs successfully on a standard CPU environment.)

II - Installation (if new to Python):
-----
Python is open source and can be downloaded from https://www.python.org/downloads/
A virtual environment can be set up with the required packages using the following command lines:

To create the environment named "test_env" using venv:
python -m venv test_env

To activate the environment:
[Windows] test_env\Scripts\activate

Install requirements:
pip install -r requirements.txt

While installation times may vary, the installation of Python and all required packages should be completed within 10–20 minutes.

III- Demos and Instructions
-----
To replicate the analyses and regenerate the figures presented in the paper, you can run the corresponding Python scripts directly from the terminal. For example:

Figure3.py
Figure4.py
Figure5.py
Figure6.py
Figure7.py

Most Python scripts (e.g., standard data processing and figure plotting) should execute within a minute. However, scripts involving deep learning evaluations or SHAP feature importance analysis (such as calculating average importance across multiple random seeds) may take a few minutes to complete depending on your hardware.

Note: Due to the inherent stochastic nature of neural network training, slight numerical variations in physics losses and network responses may occur across different runs. Consequently, the newly generated results might not be perfectly identical to the exact values reported in our manuscript. However, the qualitative trend remains highly consistent, and the superior performance of our proposed method over baseline models (FNN, CNN, and LSTM) remains robust and evident.

IV- Troubleshooting
-----
1. **PyTorch CUDA / Device Errors:**
   The code automatically detects and utilizes a GPU if available. If you encounter device mismatch errors (e.g., `RuntimeError: Expected all tensors to be on the same device`), ensure that your input data and model are both on the same device (CPU or CUDA). You can force the code to run on CPU by modifying the torch device configuration in the scripts.

2. **SHAP Plotting Display:**
   If the SHAP summary plots or heatmaps do not render text correctly, it is usually a matplotlib font compatibility issue. Ensure that `matplotlib` is updated to the version specified in `requirements.txt`.
