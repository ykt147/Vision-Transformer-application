
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None):
        self.meta = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.meta['dx'].unique()))}
        self.meta['label'] = self.meta['dx'].map(self.label_map)
        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        img_id = self.meta.iloc[idx]['image_id']
        label = self.meta.iloc[idx]['label']

        found = False
        for part in ['part_1', 'part_2']:
            img_path = os.path.join(self.img_root, part, img_id + '.jpg')
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                found = True
                break

        if not found:
            raise FileNotFoundError(f"Image {img_id} not found in {self.img_root}/part_1 or part_2")

        if self.transform:
            image = self.transform(image)

        return image, label