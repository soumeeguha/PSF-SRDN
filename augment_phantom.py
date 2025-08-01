import numpy as np 
import matplotlib.pyplot as plt 
import glob 
import random 
import scipy.ndimage as ndimage
import cv2
import os 

root = 'images/'
save = 'augmented_images/'
folders = glob.glob(root + '*')


data_segment = 3

save_folders = glob.glob(f'{save}{data_segment}/*')
# print(save_folders)



train_ratio = 0.7
num_imgs = 100 

train_indices = sorted(random.sample(range(1, num_imgs+1), int((train_ratio)*num_imgs)))
val_indices = list(set(list(range(1, num_imgs+1))) - set(train_indices))

angles = [0, 90, 180, 270]
means = [0, 0.025,  0.05]
sigmas = [0, 0.001]
num_im = 0
for f in range(len(folders)):
    print(folders[f])

    os.mkdir(f'{save_folders[f]}/train')
    os.mkdir(f'{save_folders[f]}/test')
    for i in range(num_imgs):
        im = cv2.imread(f'{folders[f]}/img_{i+1}.png', cv2.IMREAD_GRAYSCALE)/255


        if i+1 in train_indices:
            for angle in angles:
                im_new = ndimage.rotate(im, angle, reshape=False)
                for mean in means:
                    for sigma in sigmas:
                        num_im += 1
                        noise = np.random.normal(mean,sigma,(im.shape[0], im.shape[1]))
                        cv2.imwrite(save_folders[f] + f'/train/img_{num_im}.png', (im+noise)*255)
        else:
            num_im += 1
            cv2.imwrite(save_folders[f] + f'/test/img_{num_im}.png', im*255)

