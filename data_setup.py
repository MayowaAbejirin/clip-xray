from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch import optim
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import clip

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training


class ImageTitleDataset(Dataset):
    def __init__(self, image_csv_path, text_csv_path):
        # Load CSV files
        self.image_df = pd.read_csv(image_csv_path)
        self.text_df = pd.read_csv(text_csv_path)
        
        # both CSVs have a "uid" column for linking
        self.image_path = self.image_df['filename'].tolist()
        self.uid = self.image_df['uid'].tolist()
        
        # Initialize an empty list to store text data
        self.text_data = []

        # Match "uid" values and load "impression" column from text CSV
        for uid in self.uid:
            text_row = self.text_df.loc[self.text_df['uid'] == uid]
            if not text_row.empty:
                self.text_data.append(text_row.iloc[0]['image'])
            else:
                # Handle cases where there is no matching "uid"
                self.text_data.append("")  # or any other suitable placeholder
        # Tokenize text
        self.title = clip.tokenize(self.text_data)


    def __len__(self):
        return len(self.title)

#     def __getitem__(self, idx):
#         image = preprocess(Image.open(f"/home/Documents/IU/images/images_normalized/{self.image_path[idx]}"))
#         #print(image)
#         #title = clip.tokenize(self.title[idx])
#         #print(title)
#         return image, self.title
    
    
    def __getitem__(self, idx):
        image = preprocess(Image.open(f"/home/mayowa/Documents/IU/images/images_normalized/{self.image_path[idx]}"))
        title = self.title[idx]
        return image, title

# Example usage:
# image_csv_path = 'image_data.csv'
# text_csv_path = 'text_data.csv'
# dataset = ImageTitleDataset(image_csv_path, text_csv_path)
# image, title = dataset[0]
