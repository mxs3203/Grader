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

print(torch.__version__)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
    plt.savefig("plots/UMAP_{}.png".format(ep))

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

def nt_xent_loss2(embeddings, temperature):
    """
    Computes the NT-Xent loss given a batch of embeddings where each pair of embeddings are
    expected to be similar (positive pairs), and all other pairs are dissimilar (negative pairs).
    embeddings: concatenated embeddings [2*batch_size, embedding_dim] from augmentations
    temperature: a scalar for temperature scaling
    """
    batch_size = embeddings.shape[0] // 2  # Adjust if each SQL has multiple augmentations
    z = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    # Create mask to select positive and negative examples
    mask = torch.eye(batch_size * 2, device=z.device, dtype=bool)
    mask = mask.invert()

    # Select the logits where mask is True (i.e., exclude diagonals which are the positives)
    negatives = similarity_matrix[mask].view(2 * batch_size, -1)

    # Select the positives: each embedding should be similar to its augmented versions
    # (Assuming augmented versions are adjacent in the batch)
    pos_mask = torch.eye(batch_size * 2, batch_size * 2, device=z.device, dtype=bool)
    pos_mask = pos_mask.roll(shifts=batch_size, dims=0)
    positives = similarity_matrix[pos_mask].view(2 * batch_size, -1)

    # Calculating the loss
    nominator = torch.exp(positives / temperature)
    denominator = torch.sum(torch.exp(negatives / temperature), dim=1) + nominator
    all_losses = -torch.log(nominator / denominator)
    loss = torch.mean(all_losses)

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
UNK_TOKEN = vocabulary['<UNK>']
print(vocab_size)
data = pd.read_pickle("data/processed_data.pkl")
dataset = QuestionDataset(data, column_names=['studentsolution_padded', 'correctsolution_padded', 'percentWrong'],
                          device=device, vocab_size=vocab_size)
'''
    Spliting the data
'''
train_size = int(0.7 * len(dataset))  # 70% for training
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
'''
    Init everything
'''
def objective(trial):
    # Sample hyperparameters randomly within certain ranges
    num_epochs = 50
    batch_size = 512
    learning_rate = trial.suggest_float('learning_rate', 0.0004, 0.0006)
    embedding_dim = trial.suggest_int('embedding_dim', 32, 256)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    lstm_hidden_dim = trial.suggest_int('lstm_hidden_dim', 2, 16)
    temperature = trial.suggest_float('temperature', 0.01, 0.5)
    augment_percent = trial.suggest_int('augment_percent', 10, 40)
    # build data loaders
    sql_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    sql_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # Build and train the model using the sampled hyperparameters
    model = SQLComparisonModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, lstm_hidden=lstm_hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # You would continue with your training loop here
    model.train()
    if trial.should_prune():
        raise optuna.TrialPruned()
    torch.cuda.empty_cache()
    # For demonstration, assume loss is computed here (replace with actual training logic)
    loss = run_training(model, optimizer, batch_size, num_epochs, augment_percent, temperature, sql_train_loader, sql_test_loader, embedding_dim,hidden_dim,learning_rate)

    return loss

def run_training(model, optim, batch_size,num_epochs,augment_percent, temperature, sql_train_loader, sql_test_loader,embedding_dim, hidden_dim,learning_rate):
    '''
        Training loop
    '''
    best_loss = 1000
    for ep in range(num_epochs):
        model.train()
        running_loss_train = 0.0
        running_loss_valid = 0.0

        for i,batch_data in enumerate(sql_train_loader):
            correct_sql, student_sql, percent = batch_data['correct_sql'], batch_data['student_sql'], batch_data['percentWrong']
            optim.zero_grad()
            aug_correct_sql = torch.stack([augment_sql(sql,augment_percent, UNK_token=UNK_TOKEN) for sql in correct_sql]).to(device)
            aug_student_sql = torch.stack([augment_sql(sql,augment_percent, UNK_token=UNK_TOKEN) for sql in student_sql]).to(device)
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
                aug_correct_sql = torch.stack([augment_sql(sql,augment_percent, UNK_token=UNK_TOKEN) for sql in correct_sql]).to(device)
                aug_student_sql = torch.stack([augment_sql(sql,augment_percent, UNK_token=UNK_TOKEN) for sql in student_sql]).to(device)
                L_correct = model(aug_correct_sql)
                L_student = model(aug_student_sql)
                loss = nt_xent_loss(L_correct,L_student, temperature=temperature)
                running_loss_valid += loss.item()
        if ep % 5 == 0: # every n th epoch
            make_umap(ep,L_correct, L_student,batch_data, percent)
        if running_loss_valid / len(sql_test_loader) < best_loss:
            best_loss = running_loss_valid / len(sql_test_loader)
        print(f"Epoch [{ep + 1}/{num_epochs}], Train Loss: {running_loss_train / len(sql_train_loader):.4f}, "
              f"Valid Loss: {running_loss_valid / len(sql_test_loader)}")

    #     run.log({'Epoch': ep,"Train/loss": running_loss_train / len(sql_train_loader), 'Test/loss': running_loss_valid / len(sql_test_loader)})
    # wandb.finish()
    return best_loss

study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner(),
                            storage="sqlite:///db.sqlite3",
                            study_name="contrastive_optuna_with_lstm",
                            load_if_exists=True)
study.optimize(objective, n_trials=100, n_jobs=1)
with open('optuna_study_contrsative.pkl', 'wb') as file:
    pickle.dump(study, file)
df_trials = study.trials_dataframe()
df_trials.to_csv('optuna_trials.csv', index=False)