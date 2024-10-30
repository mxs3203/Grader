import pickle

import pandas as pd
import torchviz
import torch

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
for i,data in enumerate(dataset):
    y = model(data['correct_sql'])
    torchviz.make_dot(y, params=dict(list(model.named_parameters()))).render("torchviz_model", format="png")
    break