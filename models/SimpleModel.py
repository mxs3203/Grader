import torch
from torch import nn

import torch.nn.functional as F
class SQLComparisonModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, lstm_hidden):
        super(SQLComparisonModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden, num_layers=1,batch_first=True)
        self.fc = nn.Linear(lstm_hidden, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.a = nn.ReLU()

    def forward(self, sql):
        # Embed the correct and student SQL queries
        sql_emb = self.embedding(sql) # (batch_size, seq_len, embedding_dim) -> (batch_size, embedding_dim)
        _, (hidden_state, _) = self.lstm(sql_emb)
       #  # Shape: (batch_size, embedding_dim), FloatTensor
        hidden_state = hidden_state.squeeze(0)
        hidden = self.a(self.fc(hidden_state))
        hidden = self.out(hidden)
        hidden_normalized = F.normalize(hidden, p=2, dim=1)
        return hidden_normalized

    def forward_embeddings(self, sql_emb):
        # New method accepting embeddings directly
        _, (hidden_state, _) = self.lstm(sql_emb)
        hidden_state = hidden_state.squeeze(0)
        hidden = self.a(self.fc(hidden_state))
        hidden = self.out(hidden)
        hidden_normalized = F.normalize(hidden, p=2, dim=1)
        return hidden_normalized