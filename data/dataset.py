import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class WikiArtDs(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id_):
        row = self.df.iloc[id_]
        img_path = os.path.join(self.img_dir, row["filename"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        artist_label = torch.tensor(row['artist_enc'], dtype=torch.long)
        genre_label = torch.tensor(row['genre_enc'], dtype=torch.float32)

        return img, artist_label, genre_label
