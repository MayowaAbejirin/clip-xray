import torch.nn as nn
from torch import optim
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import clip

# main_df = pd.read_csv('/home/mayowa/Documents/IU/indiana_reports.csv')  

# id_df = pd.read_csv('/home/mayowa/Documents/IU/indiana_projections.csv')  

# main_df.loc[:5,["impression"]]

# # Check if column 'impression"' has missing values
# has_missing_values = main_df["impression"].isna().any()
# print(f"Column impression has missing values: {has_missing_values}")

# print(main_df["impression"].isna().sum()) 

# clean_df = main_df.dropna(subset=["impression"])

# has_missing_values = clean_df["impression"].isna().any()
# print(f"Column impression has missing values: {has_missing_values}")

# print(clean_df["impression"].isna().sum()) 

# file_path = '/home/mayowa/Documents/cleaned_reports.csv'

# clean_df.to_csv(file_path, index=False)

# print(main_df["uid"][0])



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


# Example usage: 
image_csv_path = '/home/mayowa/Documents/IU/indiana_projections.csv'
text_csv_path = '/home/mayowa/Documents/IU/indiana_reports.csv'
dataset = ImageTitleDataset(image_csv_path, text_csv_path)

dataset[0][1]

dataset[0][0]

len(dataset.text_data)


BATCH_SIZE = 8
train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE, shuffle=True) #Define your own dataloader

imagei, texti = next(iter(train_dataloader))

imagei, texti

from tqdm import tqdm

EPOCH = 1

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

# add your own code to track the training progress.

for epoch in range(EPOCH):
  for batch in tqdm(train_dataloader):
      optimizer.zero_grad()

      images,texts = batch 
    
      images= images.to(device)
      texts = texts.to(device)
    
      logits_per_image, logits_per_text = model(images, texts)

      ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

      total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
      total_loss.backward()
      if device == "cpu":
         optimizer.step()
      else : 
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)
      print(total_loss)


# torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': total_loss,
#         }, f"model_checkpoint/model_10.pt") #just change to your preferred folder/filename

# model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
# checkpoint = torch.load("model_checkpoint/model_10.pt")

# # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
# #checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
# #checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
# #checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

# model.load_state_dict(checkpoint['model_state_dict'])
