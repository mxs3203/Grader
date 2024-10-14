import re

import pandas as pd
import torch
from torch.utils.data import random_split
import wandb
from Dataset import QuestionDataset

if torch.cuda.is_available():
    print("CUDA is available! PyTorch is using GPU acceleration.")
    device = "cuda:1"
else:
    print("CUDA is not available. PyTorch is using CPU.")
    device = "cpu"

'''
    Setting up the data
'''
data = pd.read_csv("data/preprocessed_data.csv")
dataset = QuestionDataset(data, column_names=['studentsolution_padded', 'correctsolution_padded', 'percent'])

'''
    Hyperparams
'''
batch_size = 256
learning_rate = 0.0001
num_epochs = 30
print()
'''
    Spliting the data
'''
train_size = int(0.7 * len(dataset))  # 70% for training
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset,  [train_size, test_size] )
sql_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
sql_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for i,data in enumerate(sql_train_loader):
    correct_sql, student_sql, percent = data['correct_sql'], data['student_sql'], data['percent']
    print()

# run = wandb.init(
#     # set the wandb project where this run will be logged
#     project="EEGImage",
#     name="SSIM-ImageGenerator-{}-GAN-BCE".format(contrastive_output_size),
#     entity='mxs3203',
#     config={
#         "learning_rate": learning_rate,
#         "architecture": "LSTM_Contrastive_ImageGeneration",
#         "dataset": "EEG",
#         "batch_size": batch_size,
#         "hidden_size": hidden_size,
#         "num_layers": num_layers,
#         "num_epochs": num_epochs,
#         "output_size": contrastive_output_size,
#         "fine_tune": FINE_TUNE
#     }
# )
