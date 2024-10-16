import torch
from torch import nn

import torch.nn.functional as F
class SQLComparisonModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SQLComparisonModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.a = nn.ReLU()

    def forward(self, sql):
        # Embed the correct and student SQL queries
        sql_emb = self.embedding(sql) # (batch_size, seq_len, embedding_dim) -> (batch_size, embedding_dim)
        sql_emb_pooled = sql_emb.mean(dim=1)  # Shape: (batch_size, embedding_dim), FloatTensor
        hidden = self.a(self.fc(sql_emb_pooled))
        hidden = self.out(hidden)
        hidden_normalized = F.normalize(hidden, p=2, dim=1)
        return hidden_normalized