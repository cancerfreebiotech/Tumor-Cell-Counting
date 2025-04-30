import skimage
from skimage import measure
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
from cv2 import connectedComponents
from PIL import Image
import json
from tqdm import trange
import os
import argparse

from count_tumor import count_tumor
from compare_results import gather_labels
from img_tools import add_border, find_best_subimage



def parse_args():
    parser = argparse.ArgumentParser(description='Count tumor cells base on the images in the specified folder. If the input folder is\'t specified, it will be "./". ')
    parser.add_argument("-i", "--input_path", default='./', type=str, help="enter the root directory you want to count tumor")
    parser.add_argument("-o", "--output_path", default='./output/', type=str, help="enter the output directory")
    
    args = parser.parse_args()
    return args



def process_sample(path, path_B, path_G, path_R, output_path='./output'):
    # load images
    try:
        if os.path.exists(path_B):
            img_blue = cv2.imread(path_B)[:, :, 0]  # bgr
        else:
            print('blue image file doesn\'t exist:', path_B)
        if os.path.exists(path_G):
            img_green = cv2.imread(path_G)[:, :, 1]  # bgr
            assert img_blue.shape == img_green.shape
        else:
            print('green image file doesn\'t exist:', path_G)
        if os.path.exists(path_R):
            img_red = cv2.imread(path_R)[:, :, 2]  # bgr
            assert img_blue.shape == img_red.shape
        else:
            print('red image file doesn\'t exist:', path_R)
    except Exception as e:
        print('---read files failed---')
        print(path_B)
        print(path_G)
        print(path_R)
        print('-----------------------')
        print(e)
        return -1


    # if CD45 is in the filename, don't count the red marker
    if 'CD45' in path_B:
        definition = [True, True, False]  # (B,G,R)==(+,+,-)
    else:
        definition = [True, True, True]  # (B,G,R)==(+,+,+)
    
    output_image = True
    res = count_tumor(img_blue, img_green, img_red, definition, output_image=output_image)
    total_cnt, img_bluein, labels, tgt_green_idx_list, tgt_red_idx_list, result_idx_list, bin_b, bin_g, bin_r, area_blue, counts = res
    cnt_blue, cnt_green, cnt_red = counts
    if total_cnt < 0:
        return None

    # sample_name = os.path.split(os.path.split(path)[-2])[-1] + '_' + os.path.split(path)[-1]
    
    
    # output image directory
    output_path_full = os.path.join(output_path, 'full_img')
    output_path_best = os.path.join(output_path, 'best_img')

    if not os.path.exists(output_path_full):
        os.makedirs(output_path_full)
    if not os.path.exists(output_path_best):
        os.makedirs(output_path_best)
        


    # get boundary of best area
    best_i, best_j = find_best_subimage(img_bluein)

    
    # output blue, green and red images with yellow border
    print('generating images with yellow border...')
    img_blue = cv2.imread(path_B)
    img_green = cv2.imread(path_G)
    img_red = cv2.imread(path_R)
    image_merge = np.concatenate( [np.expand_dims(x, 2) for x in (img_blue[:, :, 0], img_green[:, :, 1], img_red[:, :, 2])] , 2)
    
    bin_b = bin_b.astype('uint8')
    bin_g = bin_g.astype('uint8')
    bin_r = bin_r.astype('uint8')
    img_bluein = img_bluein.astype('uint8')
    
    border_img_blue = add_border(img_blue, img_bluein)
    border_img_green = add_border(img_green, img_bluein)
    border_img_red = add_border(img_red, img_bluein)
    border_img = add_border(image_merge, img_bluein)


    # output full image with border
    output_path_B = os.path.join(output_path_full, 'blue_image.jpg')
    output_path_G = os.path.join(output_path_full, 'green_image.jpg')
    output_path_R = os.path.join(output_path_full, 'red_image.jpg')
    output_path_M = os.path.join(output_path_full, 'merged_image.jpg')
    
    cv2.imwrite(output_path_B, border_img_blue)
    cv2.imwrite(output_path_G, border_img_green)
    cv2.imwrite(output_path_R, border_img_red)
    cv2.imwrite(output_path_M, border_img)


    # output best img
    output_path_B = os.path.join(output_path_best, 'blue_image.jpg')
    output_path_G = os.path.join(output_path_best, 'green_image.jpg')
    output_path_R = os.path.join(output_path_best, 'red_image.jpg')
    output_path_M = os.path.join(output_path_best, 'merged_image.jpg')
    
    cv2.imwrite(output_path_B, img_blue[best_i:best_i+512, best_j:best_j+512])
    cv2.imwrite(output_path_G, img_green[best_i:best_i+512, best_j:best_j+512])
    cv2.imwrite(output_path_R, img_red[best_i:best_i+512, best_j:best_j+512])
    cv2.imwrite(output_path_M, image_merge[best_i:best_i+512, best_j:best_j+512])


    # output csv file and histogram of blue area
    print('generating histogram of area...')
    # pd.DataFrame.from_dict({'area': area_blue}).to_csv(os.path.join(output_path, 'area.csv'), encoding='utf-8')
    hist, bin_edges = np.histogram(area_blue, bins=list(range(0, max(area_blue)//10 * 10 + 20, 10)))
    pd.DataFrame.from_dict({'面積(pixel)': [ f'{i}~{j}' for i, j in zip(bin_edges[:-1], bin_edges[1:])], 
                            'number of connected components': hist}).to_csv(os.path.join(output_path, 'area.csv'), encoding='utf-8', index=False)
    
    _, _, patches = plt.hist(area_blue, bins=list(range(0, max(area_blue)//10 * 10 + 20, 10))) # interval of the histogram: 10
    plt.xlabel('area size (pixel)')
    plt.ylabel('num of connected components')
    for i in range(0, len(patches), 2):
        patches[i].set_facecolor('#C0C0C0')
    for i in range(1, len(patches), 2):
        patches[i].set_facecolor('#A9A9A9')
    plt.savefig(os.path.join(output_path, 'area.png'))
    plt.close()


    # output csv file and histogram of blue area (use num of cell instead of num of connected comp.)
    if len(result_idx_list) > 0: 
        result_area_blue = np.take(area_blue, result_idx_list).tolist()
        area_mean = int(np.array(np.mean(result_area_blue)))
        for i, area in enumerate(result_area_blue):
            n = area // area_mean
            if n >= 2:
                result_area_blue[i] = area // n
                result_area_blue += [area // n] * (n-1)
                if (area % area_mean / area_mean) >= 0.5:
                    result_area_blue.append(area % area_mean)
    else:
        result_area_blue = []
    

    # pd.DataFrame.from_dict({'area': area_blue}).to_csv(os.path.join(output_path, 'area.csv'), encoding='utf-8')
    hist, bin_edges = np.histogram(result_area_blue, bins=list(range(0, max(result_area_blue + [0, ])//10 * 10 + 20, 10)))
    pd.DataFrame.from_dict({'面積(pixel)': [ f'{i}~{j}' for i, j in zip(bin_edges[:-1], bin_edges[1:])], 
                            'number of Tumors cells': hist}).to_csv(os.path.join(output_path, 'area_tumor_cell.csv'), encoding='utf-8', index=False)
    
    _, _, patches = plt.hist(result_area_blue, bins=list(range(0, max(result_area_blue + [0, ])//10 * 10 + 20, 10))) # interval of the histogram: 10
    plt.xlabel('area size (pixel)')
    plt.ylabel('number of Tumors cells')
    for i in range(0, len(patches), 2):
        patches[i].set_facecolor('#C0C0C0')
    for i in range(1, len(patches), 2):
        patches[i].set_facecolor('#A9A9A9')
    plt.savefig(os.path.join(output_path, 'area_tumor_cell.png'))
    plt.close()

    
    # output counting result in csv
    pd.DataFrame.from_dict({'total': [total_cnt], 'blue': [cnt_blue], 'green': [cnt_green], 'red': [cnt_red]}).to_csv(os.path.join(output_path, 'result.csv'), index=False, encoding='utf-8')

    return total_cnt



def main(args):

    # if output dir doesn't exist, create one
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    for root, dirs, files in os.walk(args.input_path):
        # check if it has images of 3 channels
        cnt = 0
        print(files)
        for i in ['d0.JPG', 'd1.JPG', 'd2.JPG']:
            for f in files:
                print(f)
                if i in f and 'TD' in f and 'TM' not in f:
                    cnt += 1
                    prefix = f[:-6]
                    break
        if cnt != 3:
            continue

        print('processing', root)
        path_B = os.path.join(root, prefix + 'd0.JPG')
        path_G = os.path.join(root, prefix + 'd1.JPG')
        path_R = os.path.join(root, prefix + 'd2.JPG')
        
            
        sample_name = os.path.split(os.path.split(root)[-2])[-1] + '_' + os.path.split(root)[-1]
        total_cnt = process_sample(root, path_B, path_G, path_R, output_path=args.output_path)
        break



if __name__ == "__main__":
    args = parse_args()
    main(args)