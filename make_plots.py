import pickle

import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
from plotly.io import show

from Dataset import QuestionDataset
from models.SimpleModel import SQLComparisonModel
vocabulary = pickle.load(open("data/vocab.pkl", "rb"))
vocab_size = len(vocabulary)+1
UNK_TOKEN = vocabulary['<UNK>']
print(vocab_size)
data = pd.read_pickle("data/processed_data.pkl")
dataset = QuestionDataset(data, column_names=['studentsolution_padded', 'correctsolution_padded', 'percentWrong'],
                          device="cpu", vocab_size=323)

model = SQLComparisonModel(vocab_size=323, embedding_dim=180, hidden_dim=118)
losses = pd.read_csv("data/predictive_loss_for_visualize.csv")
# sns.scatterplot(losses, x='Step', y='TestLoss',color ="#006400", label = "Test Loss")
# sns.lineplot(losses, x='Step', y='TestLoss', color ="#006400")
# sns.scatterplot(losses, x='Step', y='Train/loss', color="#ADD8E6", label = "Train Loss")
# sns.lineplot(losses, x='Step', y='Train/loss', color="#ADD8E6")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# #plt.show()
# plt.savefig("plots/F3_real_training.pdf")
# plt.close()
#
# fine_tune_false = pd.read_csv("data/fine_tune_false.csv")
# sns.scatterplot(fine_tune_false, x='Step', y='Test-loss',color ="#006400", label = "Test Loss No Fine Tunning")
# sns.lineplot(fine_tune_false, x='Step', y='Test-loss', color ="#006400")
# sns.scatterplot(losses, x='Step', y='TestLoss', color="#ADD8E6", label = "Test Loss with Fine Tunning")
# sns.lineplot(losses, x='Step', y='TestLoss', color="#ADD8E6")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# #plt.show()
# plt.savefig("plots/F3_fine_tune_vs_no.pdf")
# print()

# hyperparameters = ['Temperature', 'Embedding Dim', 'Augmentation Percent', 'Hidden Dim', 'Learning Rate']
# importance_values = [0.48, 0.26, 0.16, 0.05, 0.05]
# plt.figure(figsize=(10, 6))
# sns.barplot(x=hyperparameters, y=importance_values, palette=['#003f5c'])  # Using a consistent dark blue color
# plt.xlabel('Hyperparameters')
# plt.ylabel('Importance')
# plt.title('Importance of Different Hyperparameters in the Model')
# plt.savefig("optuna/importance.pdf")



study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner(),
                            storage="sqlite:///db.sqlite3",
                            study_name="contrastive_optuna_GA3",
                            load_if_exists=True)
importance_dict = optuna.importance.get_param_importances(study)
print(importance_dict)
fig = optuna.visualization.plot_param_importances(study)
fig.write_image("optuna/plotly_paramimportance.pdf")
fig = optuna.visualization.plot_contour(study, params=["temperature", "augment_percent"])
fig.write_image("optuna/plotly_contour.pdf")
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.write_image("optuna/parrallel.pdf")
print()