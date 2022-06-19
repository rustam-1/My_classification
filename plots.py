import matplotlib.pyplot as plt

import cv2

import random
from typing import List, Tuple, Any


def show_images_path(paths: List[List[str]], resize: Tuple[int, int]=None, shuffle: Any=False, titles: List[str]=None) -> None:
    n_cols = len(paths)
    n_rows = len(paths[0])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

    if titles:
        for i, title in enumerate(titles):
            if n_cols > 1:
                axes[0, i].set_title(title)
            else:
                axes[i].set_title(title)

    if shuffle:
        random.shuffle(paths)
    
    for row in range(n_rows):
        for col in range(n_cols):
            image = cv2.imread(paths[col][row])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if resize:
                image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)

            if n_cols > 1:
                axes[row, col].imshow(image)
                axes[row, col].set_axis_off()
            else:
                axes[row].imshow(image)
                axes[row].set_axis_off()


def show_image_tensor(image, image_class: str=None, classes_names: List[str]=None, probality_classes=None, sep: str=' ') -> None:
    image = image.detach().cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
  
    if image_class and probality_classes is None:
        plt.title(image_class)
        return

    if classes_names is None:
        class_name = [i for i in range(len(probality_classes))]
        
    title = [image_class] if image_class else []
    
    for name, p in zip(classes_names, probality_classes):
        title.append(f'{name} = {round(p.item(), 3)}')
            
    plt.title(sep.join(title))


def plot_pies(parts: List[List[int]], labels: List[List[str]],
              image_paths: List[str]=None, resize: Tuple[str, str]=(100, 100),
              explodes: List[Tuple[int]]=None, col_sizes: int=6, row_sizes: int=5,
              color_dict: dict=None) -> None:
    n_rows = len(parts)
    n_cols = 1 if image_paths is None else 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * col_sizes, n_rows * row_sizes))

    for row in range(n_rows):
        parts_row = parts[row]
        labels_row = labels[row]
        explode_row = [0] * len(parts_row) if explodes is None else explodes[row]
        colors = None

        if color_dict:
            colors = [color_dict[label] for label in labels_row]

        if image_paths:
            image = cv2.imread(image_paths[row])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, resize, cv2.INTER_AREA)
        
        if n_rows == 1 and n_cols == 1:
            axes.pie(parts_row, explode=explode_row, labels=labels_row,
                      shadow=True, colors=colors)
        elif n_rows == 1 and n_cols > 1:
            axes[0].pie(parts_row, explode=explode_row, labels=labels_row,
                        shadow=True, colors=colors)
            axes[1].imshow(image)
            axes[1].set_axis_off()
        elif n_rows > 1 and n_cols == 1:
            axes[row].pie(parts_row, explode=explode_row, labels=labels_row,
                            shadow=True, colors=colors)
        else:
            axes[row, 0].pie(parts_row, explode=explode_row, labels=labels_row,
                            shadow=True, colors=colors)
            axes[row, 1].imshow(image)
            axes[row, 1].set_axis_off()




