import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import random_split
import wandb
from Dataset import QuestionDataset
import umap

from models.PredictPercentContrastive import SQLPredictorModel
from models.SimpleModel import SQLComparisonModel
def make_umap(correct_L, student_L, batch_data,distance, percent):
    umap_reducer = umap.UMAP(n_components=2, random_state=27)
    latent_representations = torch.cat((correct_L, student_L), dim=0).cpu().numpy()
    latent_2d = umap_reducer.fit_transform(latent_representations)
    mini_batch_size = np.shape(batch_data['correct_sql'])[0]
    correct_2d = latent_2d[:mini_batch_size]
    student_2d = latent_2d[mini_batch_size:]
    df = pd.DataFrame({
        'UMAP1': latent_2d[:, 0],  # First UMAP component
        'UMAP2': latent_2d[:, 1],  # Second UMAP component
        'label': ['correct'] * mini_batch_size + ['student'] * mini_batch_size,  # Label for correct/student
        'distance': list(distance.detach().cpu().numpy()) + list(distance.detach().cpu().numpy()),
        # Distance repeated for correct/student
        'percentWrong': list(percent.detach().cpu().numpy()) + list(percent.detach().cpu().numpy())
        # Percent repeated for correct/student
    })
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=correct_2d[:, 0], y=correct_2d[:, 1], color='blue', label='Correct SQL', marker='o')
    sns.scatterplot(x=student_2d[:, 0], y=student_2d[:, 1], color='red', label='Student SQL', marker='x')
    plt.title("UMAP Visualization of Latent Representations")
    # Draw lines connecting each pair of student and correct representations
    for j in range(mini_batch_size):
        correct_point = correct_2d[j]
        student_point = student_2d[j]
        plt.plot([correct_point[0], student_point[0]], [correct_point[1], student_point[1]], color='gray',
                 linestyle='--')
        mid_point = (correct_point + student_point) / 2  # Midpoint between correct and student points
        plt.text(mid_point[0], mid_point[1], f'{percent[j]:.2f}', fontsize=9, color='black', ha='center')
    plt.savefig("plots/UMAP_{}.png".format(ep))


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
vocabulary = pickle.load(open("data/vocab.pkl", "rb"))
vocab_size = len(vocabulary)+1
print(vocab_size)
data = pd.read_pickle("data/processed_data.pkl")
dataset = QuestionDataset(data, column_names=['studentsolution_padded', 'correctsolution_padded', 'percentWrong'],device=device, vocab_size=vocab_size)
'''
    Hyperparams
'''
batch_size = 256
learning_rate = 0.005733257548958954
num_epochs = 500
embedding_dim = 176
hidden_dim = 116
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
contrastive_model = torch.load("contrastive.pth").to(device)
for param in contrastive_model.parameters():
    param.requires_grad = True
model = SQLPredictorModel(contrastive_model,embedding_dim, hidden_dim).to(device)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss(reduction='mean')
torch.cuda.empty_cache()
'''
    Training loop
'''
run = wandb.init(
    # set the wandb project where this run will be logged
    project="Grader",
    entity="mxs3203",
    name="Predictive_SimpleEmbedding_Emb{}_Hid{}".format(embedding_dim, hidden_dim),
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
        correct_sql, student_sql, percent = batch_data['correct_sql'], batch_data['student_sql'], batch_data['percentWrong']
        optim.zero_grad()
        L_correct = model(correct_sql)
        L_student = model(student_sql)
        distance = torch.norm(L_correct - L_student, dim=1)
        loss = loss_fn(distance, percent)
        loss.backward()
        optim.step()
        running_loss_train += loss.item()

    with torch.no_grad():
        model.eval()
        for i,batch_data in enumerate(sql_test_loader):
            correct_sql, student_sql, percent = batch_data['correct_sql'], batch_data['student_sql'], batch_data['percentWrong']
            L_correct = model(correct_sql)
            L_student = model(student_sql)
            distance = torch.norm(L_correct - L_student, dim=1)
            loss = loss_fn(distance, percent)
            running_loss_valid += loss.item()
    if ep % 5 == 0: # every n th epoch
        make_umap(L_correct, L_student,batch_data, distance, percent)

    print(f"Epoch [{ep + 1}/{num_epochs}], Train Loss: {running_loss_train / len(sql_train_loader):.4f}, "
          f"Valid Loss: {running_loss_valid / len(sql_test_loader)}")
    run.log({'Epoch': ep,"Train/loss": running_loss_train / len(sql_train_loader), 'Test/loss': running_loss_valid / len(sql_test_loader)})
wandb.finish()
torch.save(model, "model_full.pth")
