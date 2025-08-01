# for averaging evaluation metrics for different data segments

import pandas as pd 
import numpy as np 

sigma_x = [2.0, 3.0, 4.0]
sigma_y = [1.5, 2.5, 3.5]

num_files = 3

all_df = pd.DataFrame(columns = ["timestep:", "MSE" , "PSNR", "SSIM"])
for i in range(len(sigma_x)):
    for num in range(num_files):
        data_segment = num+1 
    
        df = pd.read_csv(f'metrics/{data_segment}_sigmaX_{sigma_x[i]}_sigmaY_{sigma_y[i]}.csv')
        if num == 0:
            all_df = df[["timestep:", "MSE" , "PSNR", "SSIM"]].to_numpy()
        else:
            all_df += df[["timestep:", "MSE" , "PSNR", "SSIM"]].to_numpy()
       
    all_df /= num_files

    all_df = pd.DataFrame(data = all_df, columns = ["timestep:", "MSE" , "PSNR", "SSIM"])
    all_df.to_csv(f"metrics/Avg_sigmaX_{sigma_x[i]}_sigmaY_{sigma_y[i]}.csv", header=False)
