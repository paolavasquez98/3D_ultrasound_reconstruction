# 3D Ultrasound Reconstruction with Deep Learning

This project:
1. Generates scatterers from different geometric models - python
2. Performs beamforming on the scatterers to acquire synthetic 3D I/Q raw data (complex In phase-quadrature) - MATLAB 
3. Converts the complex raw data into a training and test dataset '.h5' format - python
4. Trains Deep Learning models for ultrasound reconstruction - python

## Requirements
### Python
Conda:
```bash
conda env create -f environment.yml
conda activate newEnv

or:
python -m venv newEnv
source newEnv/bin/activate
pip install -r requirements.txt
```
### MATLAB
- gpuSTA
- US_Toolbox