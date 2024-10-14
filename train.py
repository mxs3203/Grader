import pickle
import re

import pandas as pd
import torch
from torch import nn
from torch.utils.data import random_split
import wandb
from Dataset import QuestionDataset
from models.SimpleModel import SQLComparisonModel

if torch.cuda.is_available():
    print("CUDA is available! PyTorch is using GPU acceleration.")
    # Setup
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(128, 64).to(device)
    x = torch.rand(32, 128).to(device)
    try:
        output = model(x)
        print("Output:", output)
    except Exception as e:
        print(f"Error: {e}")
    # Move a random tensor to the GPU
    print(x)
else:
    print("CUDA is not available. PyTorch is using CPU.")
    device = "cpu"

'''
    Setting up the data
'''
data = pd.read_pickle("data/processed_data.pkl")
dataset = QuestionDataset(data, column_names=['studentsolution_padded', 'correctsolution_padded', 'percent'],device=device)
vocabulary = pickle.load(open("data/vocab.pkl", "rb"))
vocab_size = len(vocabulary)+1
print(vocab_size)
'''
    Hyperparams
'''
batch_size = 256
learning_rate = 0.0001
num_epochs = 100
embedding_dim = 256
hidden_dim = 256

'''
    Spliting the data
'''
train_size = int(0.7 * len(dataset))  # 70% for training
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset,  [train_size, test_size] )
sql_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
sql_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
'''
    Init everything
'''
model = SQLComparisonModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss(reduction='mean')
torch.cuda.empty_cache()
'''
    Training loop
'''
wandb.init(
    # set the wandb project where this run will be logged
    project="Grader",

    # track hyperparameters and run metadata
    config={
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "architecture": "Embedding+Linear",
        "dataset": "PostgresSQL",
        "epochs": num_epochs,
    }
)
for ep in range(num_epochs):

    model.train()
    running_loss_train = 0.0
    running_loss_valid = 0.0

    for i,batch_data in enumerate(sql_train_loader):
        correct_sql, student_sql, percent = batch_data['correct_sql'], batch_data['student_sql'], batch_data['percent']
        optim.zero_grad()
        distance = model(correct_sql, student_sql)
        loss = loss_fn(distance, percent)
        loss.backward()
        optim.step()
        running_loss_train += loss.item()

    with torch.no_grad():
        model.eval()
        for i,batch_data in enumerate(sql_test_loader):
            correct_sql, student_sql, percent = batch_data['correct_sql'], batch_data['student_sql'], batch_data['percent']
            distance = model(correct_sql, student_sql)
            loss = loss_fn(distance, percent)
            running_loss_valid += loss.item()

    print(f"Epoch [{ep + 1}/{num_epochs}], Train Loss: {running_loss_train / len(sql_train_loader):.4f}, "
          f"Valid Loss: {running_loss_valid / len(sql_test_loader)}")
    wandb.log({'Epoch': ep,"Train/loss": running_loss_train / len(sql_train_loader), 'Test/loss': running_loss_valid / len(sql_test_loader)})

wandb.finish()