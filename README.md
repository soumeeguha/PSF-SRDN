# PSF-SRDN

PSF-SRDN: Point Spread Function-Aware Speckle Reducing Diffusion Network

A novel diffusion model framework for reducing speckle from images which are distorted by the point spread function (PSF) of the imaging device.

Please cite: 
```
1. PSF-SRDN: Point Spread Function-Aware Speckle Reducing Diffusion Network
2. SDDPM: https://arxiv.org/abs/2311.10868

```


Phantom Dataset is provided in the folder "phantom/"

## Installation

```bash
git clone https://github.com/soumeeguha/PSF-SRDN.git
cd PSF-SRDN
conda env create -f environment.yml
conda activate psf_srdn
pip install -e .
```

### Usage
```
python main.py --root /path/to/images --epochs 150 --cuda_id 0

```

### Requirements

All dependencies are listed in requirements.txt

### Evaluation

After training, the model automatically runs evaluation on selected timesteps and PSF parameters. Results (MSE, PSNR, SSIM) are saved as CSV files in the directory specified by --save_metrics_path.

You can also customize evaluation scripts or add visualization.

### License
This project is licensed under the MIT License.

