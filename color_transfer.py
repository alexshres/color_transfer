# Alex Shrestha
# FILE: color_transfer.py
# Implementation is based on the paper Color Transfer between Images by Reinhard et al.

import cv2
import numpy as np
import sys

def convert_color_space_BGR_to_RGB(img_BGR):
    return img_BGR[:, :, ::-1].copy()

def convert_color_space_RGB_to_BGR(img_RGB):
    return img_RGB[:, :, ::-1].copy()

def convert_color_space_RGB_to_Lab(img_RGB):
    '''
    convert image color space RGB to Lab
    '''
    # Ensure input is float32 and copy to avoid modifying original
    img_RGB = img_RGB.astype(np.float32)
    
    # RGB to LMS
    rgb_lms_conv = np.array([[0.3811, 0.5783, 0.0402],
                            [0.1967, 0.7244, 0.0782],
                            [0.0241, 0.1288, 0.8444]], dtype=np.float32)
    
    # Convert to LMS space
    img_LMS = np.zeros_like(img_RGB, dtype=np.float32)
    for i in range(3):
        img_LMS[:,:,i] = (rgb_lms_conv[i,0] * img_RGB[:,:,0] + 
                         rgb_lms_conv[i,1] * img_RGB[:,:,1] + 
                         rgb_lms_conv[i,2] * img_RGB[:,:,2])
    
    # Take log of LMS values
    eps = 1e-8
    img_LMS = np.log10(img_LMS + eps)
    
    # Convert LMS to Lab
    img_Lab = np.zeros_like(img_LMS, dtype=np.float32)
    
    # L = (1/√3)(l + m + s)
    img_Lab[:,:,0] = (1.0/np.sqrt(3.0)) * (img_LMS[:,:,0] + img_LMS[:,:,1] + img_LMS[:,:,2])
    
    # a = (1/√6)(l + m - 2s)
    img_Lab[:,:,1] = (1.0/np.sqrt(6.0)) * (img_LMS[:,:,0] + img_LMS[:,:,1] - 2*img_LMS[:,:,2])
    
    # b = (1/√2)(l - m)
    img_Lab[:,:,2] = (1.0/np.sqrt(2.0)) * (img_LMS[:,:,0] - img_LMS[:,:,1])
    
    return img_Lab

def convert_color_space_Lab_to_RGB(img_Lab):
    '''
    convert image color space Lab to RGB
    '''
    img_LMS = np.zeros_like(img_Lab, dtype=np.float32)
    
    # Convert back to LMS
    # l = ((√3/3)L + (√6/6)a + (√2/2)b)
    img_LMS[:,:,0] = ((np.sqrt(3.0)/3.0) * img_Lab[:,:,0] + 
                      (np.sqrt(6.0)/6.0) * img_Lab[:,:,1] + 
                      (np.sqrt(2.0)/2.0) * img_Lab[:,:,2])
    
    # m = ((√3/3)L + (√6/6)a - (√2/2)b)
    img_LMS[:,:,1] = ((np.sqrt(3.0)/3.0) * img_Lab[:,:,0] + 
                      (np.sqrt(6.0)/6.0) * img_Lab[:,:,1] - 
                      (np.sqrt(2.0)/2.0) * img_Lab[:,:,2])
    
    # s = ((√3/3)L - (2√6/6)a)
    img_LMS[:,:,2] = ((np.sqrt(3.0)/3.0) * img_Lab[:,:,0] - 
                      (2.0*np.sqrt(6.0)/6.0) * img_Lab[:,:,1])
    
    img_LMS = 10.0 ** img_LMS
    
    # Convert LMS to RGB
    lms_rgb_conv = np.array([[4.4679, -3.5873, 0.1193],
                            [-1.2186, 2.3809, -0.1624],
                            [0.0497, -0.2439, 1.2045]], dtype=np.float32)
    
    img_RGB = np.zeros_like(img_LMS, dtype=np.float32)
    for i in range(3):
        img_RGB[:,:,i] = (lms_rgb_conv[i,0] * img_LMS[:,:,0] + 
                         lms_rgb_conv[i,1] * img_LMS[:,:,1] + 
                         lms_rgb_conv[i,2] * img_LMS[:,:,2])
    
    return img_RGB

def color_transfer(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')
    
    # Convert to Lab space
    src_lab = convert_color_space_RGB_to_Lab(img_RGB_source)
    tgt_lab = convert_color_space_RGB_to_Lab(img_RGB_target)
    
    # Calculate statistics
    src_mean = np.mean(src_lab, axis=(0,1), keepdims=True)
    tgt_mean = np.mean(tgt_lab, axis=(0,1), keepdims=True)
    
    src_std = np.std(src_lab, axis=(0,1), keepdims=True)
    tgt_std = np.std(tgt_lab, axis=(0,1), keepdims=True)
    
    # Scale the centered source data by relative standard deviations
    result_lab = ((src_lab - src_mean) * (tgt_std / (src_std + 1e-8))) + tgt_mean
    
    # Convert back to RGB
    result_rgb = convert_color_space_Lab_to_RGB(result_lab)
    
    return result_rgb

def rmse(apath,bpath):
    """
    Helper function to get RMSE score.
    apath: path to your result
    bpath: path to our reference image
    when saving your result to disk, please clip it to 0,255:
    .clip(0.0, 255.0).astype(np.uint8))
    """
    a = cv2.imread(apath).astype(np.float32)
    b = cv2.imread(bpath).astype(np.float32)
    print(np.sqrt(np.mean((a-b)**2)))


if __name__ == "__main__":
    print('==================================================')
    
    path_file_image_source = sys.argv[1]
    path_file_image_target = sys.argv[2]
    path_file_image_result = sys.argv[3]

    # img_RGB_source: is the image you want to change color
    # img_RGB_target: is the image containing the color distribution that you want to change the source image to


    source_img = cv2.imread(path_file_image_source)
    if source_img is None:
        sys.exit("Could not read the image.")

    target_img = cv2.imread(path_file_image_target)
    if target_img is None:
        sys.exit("Could not read the image.")

    src_rbg = convert_color_space_BGR_to_RGB(source_img)
    tgt_rbg = convert_color_space_BGR_to_RGB(target_img)

    rgb_result = color_transfer(src_rbg,tgt_rbg)
    bgr_result = convert_color_space_RGB_to_BGR(rgb_result)
    bgr_result = np.clip(bgr_result, 0.0, 255.0).astype(np.uint8)

    cv2.imwrite(path_file_image_result, bgr_result)

