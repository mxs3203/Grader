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

from models.AttentionBasedModel import AttentionComparisonModel
from models.PredictPercentContrastive import SQLPredictorModel
from models.SimpleModel import SQLComparisonModel


def indices_to_words(indices, vocab):
    """
    Convert a list of token indices to words using the vocabulary.

    Args:
        indices (list): List of token indices.
        vocab (dict): Mapping from token to index.

    Returns:
        list: List of words corresponding to the token indices.
    """
    inv_vocab = {v: k for k, v in vocab.items()}  # Invert the vocabulary
    return [inv_vocab.get(idx, '<UNK>') for idx in indices]  # Convert indices to words


def visualize_attention(student_sql, correct_sql, attention_scores, vocab, percent, distance):
    """
    Visualize attention scores using a heatmap. The x-axis represents tokens from correct SQL,
    and the y-axis represents tokens from student SQL.

    Args:
        student_sql (tensor): Tensor of student SQL token indices (batch_size, seq_len).
        correct_sql (tensor): Tensor of correct SQL token indices (batch_size, seq_len).
        attention_scores (tensor): Attention scores from the model (batch_size, seq_len, seq_len).
        vocab (dict): Vocabulary mapping token to index.
    """
    percent = percent.cpu().numpy()[0]
    distance = distance.cpu().numpy()[0]
    # Convert token indices to words for visualization
    student_sql_words = indices_to_words(student_sql.cpu().numpy()[0], vocab)  # First batch
    correct_sql_words = indices_to_words(correct_sql.cpu().numpy()[0], vocab)  # First batch
    max_tokens = 50

    # Convert attention scores to numpy
    attention_matrix = attention_scores.cpu().numpy()[0][:max_tokens, :max_tokens] # (seq_len, seq_len)
    attention_matrix = attention_matrix.astype(float)
    # Create a heatmap to visualize the attention scores
    plt.figure(figsize=(12, 12))
    sns.heatmap(attention_matrix, xticklabels=correct_sql_words[:max_tokens], yticklabels=student_sql_words[:max_tokens], cmap="YlGnBu",
                annot=True, fmt=".2f",annot_kws={"size": 5}, cbar_kws={"shrink": 0.75} )
    plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis tokens for better visibility
    plt.yticks(fontsize=10)
    plt.xlabel("Correct SQL Tokens")
    plt.ylabel("Student SQL Tokens")
    plt.title("Attention Scores for predicited={}, percentWrong={}".format(distance, percent))
    plt.savefig("plots/att.png")
    plt.show()

    misaligned_tokens = []

    # Iterate through the student tokens and check where their attention is focused in the correct SQL
    for i, student_token in enumerate(student_sql_words[:max_tokens]):
        # Get the attention scores for this student token across all correct SQL tokens
        attention_to_correct = attention_matrix[i]

        # Find the correct SQL token that this student token pays the most attention to
        max_attention_idx = np.argmax(attention_to_correct)
        max_attention_score = attention_to_correct[max_attention_idx]
        correct_token = correct_sql_words[max_attention_idx]

        # If the attention is high but the tokens don't match, mark this as a potential misalignment
        if max_attention_score > 0.02 and student_token != correct_token:
            misaligned_tokens.append((student_token, correct_token, max_attention_score))

    for student_token, correct_token, score in misaligned_tokens:
        print(f"Student Token: {student_token}, Correct Token: {correct_token}, Attention Score: {score:.2f}")


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
learning_rate = 0.0001
num_epochs = 150
embedding_dim = 100
hidden_dim = 64
num_heads = 10
assert embedding_dim % num_heads == 0

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

model = AttentionComparisonModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_heads=4).to(device)
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
    name="Att_Emb{}_Hid{}_Head{}".format(embedding_dim, hidden_dim, num_heads),
    # track hyperparameters and run metadata
    config={
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
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
        distance, attention_scores = model(student_sql, correct_sql)
        loss = loss_fn(distance, percent)
        loss.backward()
        optim.step()
        running_loss_train += loss.item()

    with torch.no_grad():
        model.eval()
        for i,batch_data in enumerate(sql_test_loader):
            correct_sql, student_sql, percent = batch_data['correct_sql'], batch_data['student_sql'], batch_data['percentWrong']
            distance, attention_scores = model(student_sql,correct_sql )
            loss = loss_fn(distance, percent)
            running_loss_valid += loss.item()
    if ep % 5 == 0: # every n th epoch
        visualize_attention(student_sql, correct_sql, attention_scores, vocabulary, percent, distance)

    print(f"Epoch [{ep + 1}/{num_epochs}], Train Loss: {running_loss_train / len(sql_train_loader):.4f}, "
          f"Valid Loss: {running_loss_valid / len(sql_test_loader)}")
    run.log({'Epoch': ep,"Train/loss": running_loss_train / len(sql_train_loader), 'Test/loss': running_loss_valid / len(sql_test_loader)})
wandb.finish()
torch.save(model, "model_full.pth")
