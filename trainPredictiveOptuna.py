import pickle

import numpy as np
import optuna
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

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline

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
    Spliting the data
'''
train_size = int(0.7 * len(dataset))  # 70% for training
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset,  [train_size, test_size])
'''
    Init everything
'''
def objective(trial):
    # Sample hyperparameters randomly within certain ranges
    num_epochs = 50
    batch_size = 256
    fine_tune_or_transfer = trial.suggest_categorical('fine_tune', [True, False])
    distance_measure = 'fro'#trial.suggest_categorical('distance_measure', ['fro', 'nuc'])
    learning_rate = trial.suggest_float('learning_rate', 0.003, 0.005)
    embedding_dim = 227 # must be compatible with contrastive
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    # build data loaders
    sql_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    sql_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    contrastive_model = torch.load("contrastive.pth").to(device)
    for param in contrastive_model.parameters():
        param.requires_grad = fine_tune_or_transfer

    model = SQLPredictorModel(contrastive_model, embedding_dim, hidden_dim).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    torch.cuda.empty_cache()
    # For demonstration, assume loss is computed here (replace with actual training logic)
    loss = run_training(model, loss_fn, optimizer, batch_size, num_epochs, sql_train_loader, sql_test_loader, distance_measure, trial)

    return loss

'''
    Training loop
'''
def run_training(model,loss_fn, optim, batch_size,num_epochs, sql_train_loader, sql_test_loader, distance_measure, trial):
    best_loss = 1000
    for ep in range(num_epochs):
        model.train()
        running_loss_train = 0.0
        running_loss_valid = 0.0

        for i,batch_data in enumerate(sql_train_loader):
            correct_sql, student_sql, percent = batch_data['correct_sql'], batch_data['student_sql'], batch_data['percentWrong']
            optim.zero_grad()
            L_correct = model(correct_sql)
            L_student = model(student_sql)
            distance = torch.norm(L_correct - L_student, dim=1,  p=distance_measure)
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
                distance = torch.norm(L_correct - L_student, dim=1, p=distance_measure)
                loss = loss_fn(distance, percent)
                running_loss_valid += loss.item()
        trial.report(running_loss_valid / len(sql_test_loader), ep)
        if running_loss_valid / len(sql_test_loader) < best_loss:
            best_loss = running_loss_valid / len(sql_test_loader)
        if trial.should_prune():
            raise optuna.TrialPruned()
        print(f"Epoch [{ep + 1}/{num_epochs}], Train Loss: {running_loss_train / len(sql_train_loader):.4f}, "
              f"Valid Loss: {running_loss_valid / len(sql_test_loader)}")

    return best_loss

study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=27),
                            pruner=optuna.pruners.MedianPruner(),
                            storage="sqlite:///db.sqlite3",
                            study_name="predictive_optuna")
study.optimize(objective, n_trials=50, n_jobs=1)
df_trials = study.trials_dataframe()
with open('optuna_study_object_predictive.pkl', 'wb') as file:
    pickle.dump(study, file)
df_trials.to_csv('optuna_trials_predictive.csv', index=False)