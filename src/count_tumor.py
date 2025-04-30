import skimage
from skimage import measure
from  matplotlib import pyplot as plt
import numpy as np
import cv2
from cv2 import connectedComponents
from PIL import Image
import json
from tqdm import trange
import os
import argparse



def count_tumor(image_b, image_g, image_r, definition, output_image=False):
    """
    input 3 image file(in numpy ndarray format) of a single sample and output counting result binarized image. 

    input:
        image_b: ndarray, blue image. 
        image_g: ndarray, green image. 
        image_r: ndarray, red image. 
        definition: tuple of booleans, BGR definition of the target tumor cell. e.g. if 'CD45(red)' is used, the definition (B,G,R): (True,True,False)
        output_image: bool, whether to output marked image, will increase processing time. default is `False`. 

    return:
        total_cnt, image_bin, labels, tgt_green_idx_list, tgt_red_idx_list, result_idx_list, bin_b, bin_g, bin_r, area_blue

        total_cnt: int, final count, considered clustered cells. 
        image_bin: ndarray, final masked image. 0:background ; 255:target. 
        labels: result of cv2.connectedComponents(bin_b), which is a 2D image array of the indexed connected components(1, 2, 3, ..., 0 represent the background) of BLUE IMAGE. note that green and red image are not considered. 
        tgt_green_idx_list: list of ids of connected components that satisfy the constraint of green marker. 
        tgt_red_idx_list: list of ids of connected components that satisfy the constraint of red marker. 
        result_idx_list: list of ids of connected components that satisfy the constraint of target tumor cell. 
        result: list of int, indices of target connected components. 
        bin_b: ndarray, binarized blue image. 0 or 255. 
        bin_g: ndarray, binarized green image. 0 or 255. 
        bin_r: ndarray, binarized red image. 0 or 255. 
        area_blue: list of blue areas
    """

    print(definition)
    # processing blue marker
    image_b = cv2.medianBlur(image_b, 5)
    print('binarizing...')
    bin_b = cv2.adaptiveThreshold(image_b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -16)
    print('counting connecting components...', end='')
    ret, labels = cv2.connectedComponents(bin_b)
    cnt = ret-1
    if cnt == 0:
        return -1, None, None, None, None, None, None, None
    area_blue = np.bincount(labels.flat)[1:]
    print(cnt)



    # processing green marker
    # thresholding
    image_g = cv2.medianBlur(image_g, 5)
    image_g = cv2.GaussianBlur(image_g, (5, 5), 0)
    bin_g = cv2.adaptiveThreshold(image_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, -8)

    mask_g = (bin_g == 255)
    ma_g = np.ma.array(image_g, mask=mask_g)
    thres = int(ma_g.mean()) + 8
    print(f'green thres: {thres}')
    _, bin_g = cv2.threshold(image_g, thres, 255, cv2.THRESH_BINARY)
    bin_g_bool = bin_g // 255

    brightness_green = np.bincount(labels.flat, weights=bin_g_bool.flat)[1:]
    density_green = [brightness/area for brightness, area in zip(brightness_green, area_blue)]
    tgt_green = [idx for idx, d in enumerate(density_green) if d > 0]
    print(f'green target: {len(tgt_green)} / {cnt}')



    # processing red marker
    # thresholding
    image_r = cv2.medianBlur(image_r, 5)
    image_r = cv2.GaussianBlur(image_r, (5, 5), 0)
    bin_r = cv2.adaptiveThreshold(image_r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, -8)
    mask_r = (bin_r == 255)
    ma_r = np.ma.array(image_r, mask=mask_r)
    thres = int(ma_r.mean()) + 8
    print(f'red thres: {thres}')
    _, bin_r = cv2.threshold(image_r, thres, 255, cv2.THRESH_BINARY)
    bin_r_bool = bin_r // 255

    brightness_red = np.bincount(labels.flat, weights=bin_r_bool.flat)[1:]
    density_red = [brightness/area for brightness, area in zip(brightness_red, area_blue)]
    if definition[2]:
        tgt_red = [idx for idx, d in enumerate(density_red) if d > 0]
    else:
        tgt_red = [idx for idx, d in enumerate(density_red) if d == 0]

    print(f'red target: {len(tgt_red)} / {cnt}')



    # final result
    result_idx_list = []
    for idx in range(cnt):
        if (idx in tgt_green) and (idx in tgt_red):
            result_idx_list.append(idx)
    tgtset = set(result_idx_list)

    # count clustered cells
    bonus_blue = 0
    bonus_red = 0
    bonus_green = 0
    bonus_target = 0
    area_mean = None

    if len(result_idx_list) > 0:
        area_mean = int(np.mean(np.take(area_blue, result_idx_list)))
        for idx, area in enumerate(area_blue):
            bonus = 0
            if area // area_mean >= 2:
                bonus += area // area_mean -1
                if (area % area_mean / area_mean) >= 0.5:
                    bonus += 1

            if idx in tgtset:
                bonus_target += bonus
            if idx in tgt_red:
                bonus_red += bonus
            if idx in tgt_green:
                bonus_green += bonus
            bonus_blue += bonus


    cnt_blue = cnt + bonus_blue
    cnt_green = len(tgt_green) + bonus_green
    cnt_red = len(tgt_red) + bonus_red
    total_cnt = len(result_idx_list) + int(bonus_target)


    print('avg. cell size:', area_mean)
    print('clustered:', bonus_target)
    print('final count:')

    print('    blue:', cnt_blue)
    print('    green:', cnt_green)
    print('    red:', cnt_red)
    print('    total:', total_cnt)


    image_bin = None
    if output_image:
        print('generating masked image...')
        for i in trange(labels.shape[0], ascii=True):
            for j in range(labels.shape[1]):
                if labels[i, j] != 0 and labels[i, j]-1 not in tgtset:
                    labels[i, j] = 0
        image_mask = labels > 0
        image_mask_bgr = np.repeat(np.expand_dims(image_mask, 2), 3, axis=2)
        image_bin = image_mask_bgr * 255


    tgt_green_idx_list = tgt_green
    tgt_red_idx_list = tgt_red
    counts = cnt_blue, cnt_green, cnt_red
    return total_cnt, image_bin, labels, tgt_green_idx_list, tgt_red_idx_list, result_idx_list, bin_b, bin_g, bin_r, area_blue, counts