import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

import cv2
from PIL import Image

from tqdm import tqdm
from typing import List, Tuple, Any
from collections import Counter


def define_median_size(paths: List[List[str]]) -> Tuple[int, int]:
    all_paths = []

    for part in paths:
        for class_ in part:
            all_paths.extend(class_)

    sizes = []

    for path in tqdm(all_paths):
        with Image.open(path) as f:
            sizes.append(f.size)

    print('Number of unique sizes is:', len(np.unique(sizes)))
    print('Median size is:', np.median(sizes, 0))
    print('Sizes are:', Counter(sizes))

    sizes = np.array(sizes)

    if len(np.unique(sizes)) != 1:
        sns.scatterplot(x=sizes[:, 0], y=sizes[:, 1], s=500, color=".95")
        sns.histplot(x=sizes[:, 0], y=sizes[:, 1], bins=40, pthresh=.001, cmap="mako")
        plt.xlabel('Width')
        plt.ylabel('Height')

    return tuple(np.median(sizes, 0).astype(np.int64))
   

def check_intensity(paths_list: List[List[str]], resize: Tuple[int, int]=(100, 100)) -> None:
    intensity = []

    paths = []
    for path in paths_list:
        paths.extend(path)

    for path in tqdm(paths):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
        intensity.append(image.sum())
    
    intensity = np.array(intensity)
    print('Number of black images is:', len(intensity) - sum(intensity > 0))
    print('Number of white images is:', sum(intensity == (255 * resize[0] * resize[1])))

    sns.lineplot(x=range(len(intensity)), y=sorted(intensity))
    plt.xlabel('Images')
    plt.ylabel('Intensity')


def softmax(pred):
    '''Just torch prediction'''
    
    return torch.exp(pred) / torch.sum(torch.exp(pred))