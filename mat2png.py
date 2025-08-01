import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import glob 
import os

# Load the .mat file
root = '/mnt/data_new/data_soumee/SDDPM_ultrasound/Data for Soumee/'
save_path = '/mnt/data_new/data_soumee/SDDPM_ultrasound/Data for Soumee/images/'
files = glob.glob(f'{root}*.mat')

for i in range(len(files)):
    file = files[i]
    name = file.split('/')[-1].split('.')[0]
    os.mkdir(f'{save_path}{name}')
    print(f'{name} folder created')

    data = scipy.io.loadmat(f'{root}{name}.mat')
    img = data["img"]
    for j in range(100):
        plt.imsave(f'{save_path}{name}/img_{j+1}.png', img[:, :, j])
