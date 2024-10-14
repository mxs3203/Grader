import torch
from torch import nn


class SQLComparisonModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SQLComparisonModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)  # Example pooling layer
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.a = nn.ReLU()

    def forward(self, correct_sql, student_sql):
        # Embed the correct and student SQL queries
        correct_emb = self.embedding(correct_sql).mean(dim=1) # (batch_size, seq_len, embedding_dim) -> (batch_size, embedding_dim)
        student_emb = self.embedding(student_sql).mean(dim=1) # (batch_size, seq_len, embedding_dim) -> (batch_size, embedding_dim)

        hidden_correct = self.a(self.fc(correct_emb))
        hidden_student = self.a(self.fc(student_emb))

        hidden_correct = self.a(self.out(hidden_correct))
        hidden_student = self.a(self.out(hidden_student))

        # Compute the distance between embeddings (e.g., L2 distance)
        distance = torch.norm(hidden_correct - hidden_student, dim=1)

        return distance