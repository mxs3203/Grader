
import torch

class SQLAttentionComparisonModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_heads):
        super(SQLAttentionComparisonModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.attention = torch.nn.MultiheadAttention(embedding_dim, n_heads)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, correct_sql, student_sql):
        # Embed the correct and student SQL queries
        correct_emb = self.embedding(correct_sql).transpose(0, 1)  # (seq_len, batch_size, embedding_dim)
        student_emb = self.embedding(student_sql).transpose(0, 1)  # (seq_len, batch_size, embedding_dim)

        # Apply attention to both correct and student embeddings
        correct_emb_attn, _ = self.attention(correct_emb, correct_emb, correct_emb)
        student_emb_attn, _ = self.attention(student_emb, student_emb, student_emb)

        # Pooling to get a single vector representation
        correct_emb_pooled = self.pooling(correct_emb_attn.transpose(1, 2)).squeeze(-1)
        student_emb_pooled = self.pooling(student_emb_attn.transpose(1, 2)).squeeze(-1)

        # Compute the distance between embeddings (e.g., L2 distance)
        distance = torch.norm(correct_emb_pooled - student_emb_pooled, dim=1)

        return distance
