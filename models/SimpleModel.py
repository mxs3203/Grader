import torch

class SQLComparisonModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SQLComparisonModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)  # Example pooling layer

    def forward(self, correct_sql, student_sql):
        # Embed the correct and student SQL queries
        correct_emb = self.embedding(correct_sql)  # (batch_size, seq_len, embedding_dim)
        student_emb = self.embedding(student_sql)  # (batch_size, seq_len, embedding_dim)

        # Pooling to get a single vector representation
        correct_emb_pooled = self.pooling(correct_emb.transpose(1, 2)).squeeze(-1)  # (batch_size, embedding_dim)
        student_emb_pooled = self.pooling(student_emb.transpose(1, 2)).squeeze(-1)  # (batch_size, embedding_dim)

        # Compute the distance between embeddings (e.g., L2 distance)
        distance = torch.norm(correct_emb_pooled - student_emb_pooled, dim=1)

        return distance