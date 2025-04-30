import cv2
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt


def add_border(original, binary, show=False):
    """
    add yellow border to the image `original`
    """

    original = original.astype(dtype=np.uint8)
    binary = binary.astype(dtype=np.uint8)

    # get edge mask (2D, edge:255; else:0) by using Canny algo.
    edges = cv2.Canny(binary, 100, 200)
    
    # make it a 3D ndarray
    edges = np.expand_dims(edges, 2)  # shape: (height, weight, 1), dtype: uint
    edges_mask = np.tile(edges, (1, 1, 3)) == 255  # shape: (height, weight, 1), dtype: bool
    edges_mask = np.logical_not(edges_mask)
    
    # make yellow borders
    yellow_edges = np.concatenate((np.zeros(edges.shape, dtype=np.uint8), edges, edges), 2)
    
    # clear the original pixels of the border and fill it with yellow border
    edges = original * edges_mask + yellow_edges
    

    if show:
        plt.subplot(121)
        plt.imshow(original[:, :, ::-1])
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122)
        plt.imshow(edges[:, :, ::-1])
        plt.title('Edge Image')
        plt.xticks([])
        plt.yticks([])

        plt.show()

    return edges



def find_best_subimage(image_bin):
    """
    output the boundary of the best 512*512 image, i.e. if subimage img[i:i+512, j:j+512] has largest total area of target tumor cells, output boundary i, j. 
 
    input: 
        binary image (numpy ndarray with value = 0 or 255)
    output: 
        i: y starting coordinate
        j: x starting coordinate
    """

    best_i = -1
    best_j = -1
    area_max = -1


    image_bin = image_bin // 255
    print('finding best area...')
    for i in trange(image_bin.shape[0] // 512, ascii=True):
        for j in range(image_bin.shape[1] // 512):
            area = np.sum(image_bin[i:i+512, j:j+512])
            if area > area_max:
                best_i = i
                best_j = j

    assert best_i >= 0 and best_j >= 0
    return best_i, best_j



if __name__ == "__main__":
    binary = cv2.imread('blue_binary_12_12.jpg', 0)
    original = cv2.imread('blue_12_12.jpg')
    add_border(original, binary, show=True)