import pickle

import pandas as pd
import torch
from captum.attr import IntegratedGradients, configure_interpretable_embedding_layer
from matplotlib import pyplot as plt
from Dataset import QuestionDataset
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


data = pd.read_pickle("data/processed_data.pkl")
dataset = QuestionDataset(data, column_names=['studentsolution_padded', 'correctsolution_padded', 'percentWrong'],device=device)
sql_test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
vocabulary = pickle.load(open("data/vocab.pkl", "rb"))
vocab_size = len(vocabulary)+1
print(vocab_size)

model = torch.load("./model_full.pth").to(device)
model.eval()

interpretable_embedding = configure_interpretable_embedding_layer(model, 'embedding')
ig = IntegratedGradients(model)

# Example input: student_sql and correct_sql (already tokenized and converted to tensors)
# Assuming correct_sql and student_sql are on the correct device (e.g., CUDA)

# Compute integrated gradients for the student SQL with respect to the distance
# Pass correct_sql as a "baseline" for comparison

for batch_data in sql_test_loader:
    correct_sql = batch_data['correct_sql'].to(device).long()  # Convert to LongTensor
    student_sql = batch_data['student_sql'].to(device).long()  # Convert to LongTensor
    percent_wrong = batch_data['percentWrong'].to(device)
    print(correct_sql.dtype)

    attributions = ig.attribute(inputs=(student_sql, correct_sql),
                                       baselines=(correct_sql, correct_sql),
                                       target=0,  # The target output to explain (distance between embeddings)
                                       return_convergence_delta=False)

    # Summarize the attribution for each token (you can take the L2 norm across embedding dimensions)
    attributions = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
    # Normalize the attributions for better visualization
    normalized_attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

# Create a bar plot of the attributions
plt.figure(figsize=(10, 6))
plt.bar(student_tokens, normalized_attributions)
plt.title("Token Attributions (Integrated Gradients) for Student SQL Query")
plt.ylabel("Attribution Score")
plt.xlabel("SQL Tokens")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
