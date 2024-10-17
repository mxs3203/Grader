import pickle

import numpy as np
import pandas as pd
import torch
from matplotlib import colors, pyplot as plt
from torch import nn

from Dataset import QuestionDataset

if torch.cuda.is_available():
    print("CUDA is available! PyTorch is using GPU acceleration.")
    # Setup
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(128, 64).to(device)
    x = torch.rand(32, 128).to(device)
    try:
        output = model(x)
    except Exception as e:
        print(f"Error: {e}")
else:
    print("CUDA is not available. PyTorch is using CPU.")
    device = "cpu"


def visualize_token_impact(student_sql_words, correct_sql_words, changes, title="Impact of Tokens on Student SQL",
                           cutoff=0.1):
    """
    Visualize tokens with black for low impact and red for high impact based on a cutoff value.
    Display both correct and student SQL side by side.

    Args:
        student_sql_words (list): List of tokens (words) from the student SQL.
        correct_sql_words (list): List of tokens (words) from the correct SQL.
        changes (list): List of impact values corresponding to the student SQL tokens.
        title (str): Title for the plot.
        cutoff (float): Cutoff value for considering high vs low impact.
        max_tokens_per_line (int): Maximum number of tokens per line for better readability.
    """
    # Ensure that the student_sql_words and changes have the same length
    assert len(student_sql_words) == len(changes), "Mismatch between student SQL words and impact values"
    # Create a figure and axis
    plt.figure(figsize=(20, 8))
    ax = plt.gca()
    ax.set_axis_off()
    max_tokens = 20
    rows = 0
    # Plot Correct SQL (no color coding, just plain text)
    x, y = -0.1, 0.8  # Starting position for correct solution
    for i, token in enumerate(correct_sql_words):
        if token != "<PAD>":
            if i % max_tokens == 0 and i > 0:
                x = 0  # Reset x for new line
                y -= 0.04  # Move down to next line
            plt.text(x, y, token, fontsize=12, color='black', weight='bold')  # Correct solution in black
            x += 0.05  # Move x position based on token length

    x, y = -0.1, y - 0.2  # Start student SQL below the correct solution
    for i, (token, impact) in enumerate(zip(student_sql_words, changes)):
        if token != "<PAD>":
            if i % max_tokens == 0 and i > 0:
                x = 0  # Reset x for new line
                y -= 0.04  # Move down to next line
            color = 'red' if impact >= cutoff else 'black'
            plt.text(x, y, token, fontsize=12, color=color, weight='bold')
            x += 0.05  # Move x position based on token length

    # Set title and show plot
    plt.title(title, fontsize=16)
    plt.show()

def indices_to_words(indices, vocab):
    # Invert the vocabulary to map indices back to words
    inv_vocab = {index: word for word, index in vocab.items()}

    # Convert indices to words using the inverted vocabulary
    words = [inv_vocab.get(index, '<UNK>') for index in indices]

    return words
def perturbation_analysis(correct_sql, student_sql, model, UNK_TOKEN):
    base_latent_correct = model(correct_sql)
    base_latent_student = model(student_sql)
    base_distance = torch.norm(base_latent_correct - base_latent_student, dim=1)

    distance_changes = []
    for i in range(student_sql.size(1)):  # Iterate over the token positions
        perturbed_student = student_sql.clone()
        perturbed_student[:, i] = UNK_TOKEN  # Replace token with UNKNOWN token

        perturbed_latent_student = model(perturbed_student)
        new_distance = torch.norm(base_latent_correct - perturbed_latent_student, dim=1)

        change_in_distance = abs((new_distance - base_distance).mean().item())
        distance_changes.append(change_in_distance)

    return distance_changes, base_distance

def tokens_to_dataframe(tokens_student, tokens_correct, impacts):

    data = {
        'student token': tokens_student,
        'correct token': tokens_correct,
        'impact': impacts
    }
    return pd.DataFrame(data)
vocabulary = pickle.load(open("data/vocab.pkl", "rb"))
vocab_size = len(vocabulary)+1
UNK_TOKEN = vocabulary['<UNK>']
print(vocab_size)
data = pd.read_pickle("data/processed_data.pkl")
dataset = QuestionDataset(data, column_names=['studentsolution_padded', 'correctsolution_padded', 'percentWrong'],device=device, vocab_size=vocab_size)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
model = torch.load("model_full.pth")
model.eval()
for i, batch_data in enumerate(loader):
    correct_sql, student_sql, percent = batch_data['correct_sql'], batch_data['student_sql'], batch_data['percentWrong'].detach()
    correct_sql_words = [indices_to_words(indices, vocabulary) for indices in correct_sql.detach().cpu().numpy()]
    student_sql_words = [indices_to_words(indices, vocabulary) for indices in student_sql.detach().cpu().numpy()]

    changes,base_distance = perturbation_analysis(correct_sql, student_sql, model,UNK_TOKEN)
    cutoff = (base_distance * np.median(np.abs(changes)))
    token_impact_dict = tokens_to_dataframe(student_sql_words[0],correct_sql_words[0], changes)

    visualize_token_impact(student_sql_words[0], correct_sql_words[0], changes,
                           title="Impact of Changes on Student SQL True points {} - Predicted {}".format(percent.item(),
                                                                                                         base_distance.item().__round__(3)),
                           cutoff=cutoff)
    print(f"Distance changes after perturbation: {changes}")