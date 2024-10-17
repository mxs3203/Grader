import torch
from torch import nn

import torch.nn.functional as F
class SQLPredictorModel(torch.nn.Module):
    def __init__(self, contrastive_model, embedding_dim, hidden_dim):
        super(SQLPredictorModel, self).__init__()
        self.contrastive_model = contrastive_model
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.a = nn.ReLU()

    def forward(self, sql):
        # Embed the correct and student SQL queries
        sql = self.contrastive_model(sql)
        hidden = self.a(self.fc(sql))
        hidden = self.out(hidden)
        hidden_normalized = F.normalize(hidden, p=2, dim=1)
        return hidden_normalized