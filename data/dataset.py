import os
import cv2
import numpy as np
import pandas as pd
import torch.utils.data as tordata

class DataSet(tordata.Dataset):
    def __init__(self, csv_path, root_dir):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        
        self.image_paths = self.df['Path'].values
        self.label_list = self.df['ClassId'].astype(int).values

        self.label_set = list(np.unique(self.label_list))
        self.indices_dict = {label: [] for label in self.label_set}
        for i, label in enumerate(self.label_list):
            self.indices_dict[label].append(i)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        full_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = cv2.imread(full_path)

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (64, 64))
        else:
            raise FileNotFoundError(f"Fail to Load Image: {full_path}")
        
        return image.transpose(2,0,1), self.label_list[idx]