import pickle
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from tqdm import tqdm
from Dataset import QuestionDataset
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print("CUDA is available! PyTorch is using GPU acceleration.")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# install numpy 1.26.4
else:
    print("CUDA is not available. PyTorch is using CPU.")
    device = "cpu"
def indices_to_words(indices, vocab):
    # Invert the vocabulary to map indices back to words
    inv_vocab = {index: word for word, index in vocab.items()}

    # Convert indices to words using the inverted vocabulary
    words = [inv_vocab.get(index, '<UNK>') for index in indices]

    return words
def tokens_to_dataframe(tokens_student, tokens_correct, impacts):

    data = {
        'student token': tokens_student,
        'correct token': tokens_correct,
        'impact': impacts
    }
    return pd.DataFrame(data)

def visualize_attribution(res):
    colors = ['red' if attr > 0 else 'green' for attr in res['att']]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(res['token'])), res['att'], tick_label=res['token'], color=colors)
    plt.xlabel('Tokens')
    plt.ylabel('Attribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    for bar, attr in zip(bars, res['att']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{attr:.2f}',
                 ha='center', va='bottom' if attr > 0 else 'top', color='black')

    #plt.show()
vocabulary = pickle.load(open("data/vocab.pkl", "rb"))
vocab_size = len(vocabulary)+1
UNK_TOKEN = vocabulary['<UNK>']
print(vocab_size)
data = pd.read_pickle("data/processed_data.pkl")
dataset = QuestionDataset(data, column_names=['studentsolution_padded', 'correctsolution_padded', 'percentWrong'],device=device, vocab_size=vocab_size)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
model = torch.load("model_full.pth")
model.train()
results = []
for i, batch_data in tqdm(enumerate(loader), total=len(dataset)):
    correct_sql, student_sql, percent = batch_data['correct_sql'], batch_data['student_sql'], batch_data['percentWrong'].detach()
    correct_sql_words = [indices_to_words(indices, vocabulary) for indices in correct_sql.detach().cpu().numpy()]
    student_sql_words = [indices_to_words(indices, vocabulary) for indices in student_sql.detach().cpu().numpy()]
    L_correct = model(correct_sql)
    L_student = model(student_sql)
    distance = torch.norm(L_correct - L_student, dim=1)
    student_sql_indices = student_sql  # Assuming this is a tensor of token indices
    student_embeddings = model.contrastive_model.embedding(student_sql_indices)
    student_embeddings.requires_grad_()

    correct_sql_indices = correct_sql
    correct_embeddings = model.contrastive_model.embedding(correct_sql_indices)
    correct_embeddings.requires_grad_()

    def forward_func(student_emb):
        student_output = model.forward_embeddings(student_emb)
        correct_output = model.forward_embeddings(correct_embeddings)
        distance = torch.norm(student_output - correct_output, p=2, dim=1)
        return distance


    ig = IntegratedGradients(forward_func)
    baseline_embeddings = model.contrastive_model.embedding(correct_sql)

    # Compute attributions
    attributions, delta = ig.attribute(student_embeddings, baselines=baseline_embeddings, target=None,
                                       return_convergence_delta=True)
    attributions_per_token = attributions.sum(dim=2).squeeze(0).detach().cpu().numpy()
    res = pd.DataFrame({'token':student_sql_words[0], 'att':attributions_per_token*distance.detach().cpu().numpy()})
    res = res[res['token'] != '<PAD>']
    res_sorted = res.sort_values(by='att', ascending=False)
    top_5 = res_sorted.head(5)[['token', 'att']].values.tolist()
    bottom_5 = res_sorted.tail(5)[['token', 'att']].values.tolist()
    visualize_attribution(res)
    result = {
        'student_full_solution': ' '.join([token for token in student_sql_words[0] if token != '<PAD>']),  # Full student SQL solution as a string
        'correct_full_solution': ' '.join([token for token in correct_sql_words[0] if token != '<PAD>']),  # Full correct SQL solution as a string
        'predicted_distance': distance.item(),  # Predicted distance
        'percentWrong': percent.item(),  # Ground truth percentWrong
        'student_tokens_why_wrong': top_5,
        'student_tokens_why_correct': bottom_5,
    }
    results.append(result)


# Convert the list of results into a DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("graded_ig.csv", index=False)