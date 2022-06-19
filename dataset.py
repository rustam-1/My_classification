import torch
from torch.utils.data import Dataset

import cv2

from typing import List, Tuple, Any


class People_dataset(Dataset):
    def __init__(self, paths: List[List[str]], classes: List[str], resize: Tuple[int, int]=None, transform=None):
        self.classes = classes
        self.resize = resize
        self.transform = transform

        self.paths = []
        self.targets = []
        
        for i, list_path in enumerate(paths):
            self.paths.extend(list_path)
            self.targets.extend([classes[i]] * len(list_path))

    
    def __len__(self):
        return len(self.paths)


    def __getitem__(self, idx):
        path = self.paths[idx]

        image = cv2.imread(path, cv2.IMREAD_COLOR) 
        target = self.classes.index(self.targets[idx])

        if self.resize:
            image = cv2.resize(image, self.resize, interpolation=cv2.INTER_AREA)
        
        if self.transform:
            image = self.transform(image=image)['image'].to(torch.float32)

        return image / 255.0, target