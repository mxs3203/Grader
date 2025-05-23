import pickle
import random

import optuna
import torch.nn.functional as F
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
from models.SimpleModel import SQLComparisonModel
def make_umap(ep,correct_L, student_L, batch_data, percent):
    umap_reducer = umap.UMAP(n_components=2, random_state=27)
    latent_representations = torch.cat((correct_L, student_L), dim=0).cpu().numpy()
    latent_2d = umap_reducer.fit_transform(latent_representations)
    mini_batch_size = np.shape(batch_data['correct_sql'])[0]
    correct_2d = latent_2d[:mini_batch_size]
    student_2d = latent_2d[mini_batch_size:]

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
    plt.savefig("plots/UMAP_{}.pdf".format(ep))

"""
Simple augmentation for SQL: randomly mask out some tokens.
You can implement more sophisticated augmentations if necessary.
"""
def augment_sql(sql_tensor, augment_percent,UNK_token=323):

    sql_tensor = sql_tensor.clone()  # Ensure we're not modifying the original data
    num_tokens = len(sql_tensor)

    num_masked = max(1, num_tokens // augment_percent)
    mask_indices = random.sample(range(num_tokens), num_masked)

    # Mask the selected tokens
    sql_tensor[mask_indices] = UNK_token  # Simple masking

    return sql_tensor

def device_as(t1, t2):
   """
   Moves t1 to the device of t2
   """
   return t1.to(t2.device)
def nt_xent_loss(proj_1, proj_2, temperature):
    """
    proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
    where corresponding indices are pairs
    z_i, z_j in the SimCLR paper
    """
    batch_size = proj_1.shape[0]
    z_i = F.normalize(proj_1, p=2, dim=1)
    z_j = F.normalize(proj_2, p=2, dim=1)
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)

    positives = torch.cat([sim_ij, sim_ji], dim=0)
    nominator = torch.exp(positives / temperature)

    mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
    denominator = device_as(mask, similarity_matrix) * torch.exp(similarity_matrix / temperature)

    all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
    loss = torch.sum(all_losses) / (2 * batch_size)
    return loss

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
UNK_TOKEN = vocabulary['<UNK>']
data = pd.read_pickle("data/processed_data.pkl")
dataset = QuestionDataset(data, column_names=['studentsolution_padded', 'correctsolution_padded', 'percentWrong'],device=device,vocab_size=vocab_size)
'''
    Hyperparams
'''
batch_size = 512
learning_rate = 0.0004734638139575756
num_epochs = 50
embedding_dim = 93
hidden_dim = 30
temperature = 0.4
augment_percent = 30
lstm_hidden = 1
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
model = SQLComparisonModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, lstm_hidden=lstm_hidden).to(device)
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
# You would continue with your training loop here
model.train()

torch.cuda.empty_cache()
'''
    Training loop
'''
run = wandb.init(
    # set the wandb project where this run will be logged
    project="Grader",
    entity="mxs3203",
    name="ContrastiveSimpleEmbedding_Emb{}_Hid{}".format(embedding_dim, hidden_dim),
    # track hyperparameters and run metadata
    config={
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "architecture": "Embedding+Linear",
        "dataset": "PostgresSQL",
        "epochs": num_epochs,
        "temperature": temperature,
        "augment_percent":augment_percent
    }
)
for ep in range(num_epochs):
    model.train()
    running_loss_train = 0.0
    running_loss_valid = 0.0

    for i,batch_data in enumerate(sql_train_loader):
        correct_sql, student_sql, percent = batch_data['correct_sql'], batch_data['student_sql'], batch_data['percentWrong']
        optim.zero_grad()
        aug_correct_sql = torch.stack([augment_sql(sql,augment_percent, UNK_TOKEN) for sql in correct_sql]).to(device)
        aug_student_sql = torch.stack([augment_sql(sql,augment_percent, UNK_TOKEN) for sql in student_sql]).to(device)
        L_correct = model(aug_correct_sql)
        L_student = model(aug_student_sql)
        loss = nt_xent_loss(L_correct,L_student, temperature=temperature)
        loss.backward()
        optim.step()
        running_loss_train += loss.item()

    with torch.no_grad():
        model.eval()
        for i,batch_data in enumerate(sql_test_loader):
            correct_sql, student_sql, percent = batch_data['correct_sql'], batch_data['student_sql'], batch_data['percentWrong']
            aug_correct_sql = torch.stack([augment_sql(sql,augment_percent, UNK_TOKEN) for sql in correct_sql]).to(device)
            aug_student_sql = torch.stack([augment_sql(sql,augment_percent, UNK_TOKEN) for sql in student_sql]).to(device)
            L_correct = model(aug_correct_sql)
            L_student = model(aug_student_sql)
            loss = nt_xent_loss(L_correct,L_student, temperature=temperature)
            running_loss_valid += loss.item()
    if ep % 5 == 0: # every n th epoch
        make_umap(ep,L_correct, L_student,batch_data, percent)

    print(f"Epoch [{ep + 1}/{num_epochs}], Train Loss: {running_loss_train / len(sql_train_loader):.4f}, "
          f"Valid Loss: {running_loss_valid / len(sql_test_loader)}")
    run.log({'Epoch': ep,"Train/loss": running_loss_train / len(sql_train_loader), 'Test/loss': running_loss_valid / len(sql_test_loader)})
wandb.finish()
torch.save(model, "contrastive.pth")
