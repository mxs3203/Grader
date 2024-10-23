import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionComparisonModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads=2):
        super(AttentionComparisonModel, self).__init__()
        # Embedding layer to convert tokens to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # MultiheadAttention layer
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

        # Linear layers to process the attention output
        self.fc = nn.Linear(embedding_dim*num_heads*180, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)  # Single output for attention-weighted sum

    def forward(self, student_sql, correct_sql):
        # Embed the SQL queries
        student_emb = self.embedding(student_sql)  # (batch_size, seq_len, embedding_dim)
        correct_emb = self.embedding(correct_sql)  # (batch_size, seq_len, embedding_dim)

        # Compute attention where student_emb queries correct_emb
        attention_output, attention_scores = self.attention(student_emb, correct_emb,
                                                            correct_emb)

        #student_context = attention_output.mean(dim=1)  # (batch_size, embedding_dim)
        batch_size, seq_len, embed_dim = attention_output.shape

        # Reshaping to (batch_size, seq_len, num_heads * embedding_dim)
        concatenated_attention = attention_output.reshape(batch_size, seq_len, -1)  # Concatenates all heads

        # Optional: If you want a fixed-size representation, you can pool across the sequence length
        concatenated_attention = concatenated_attention.reshape(batch_size, -1)
        # Pass the context vector through linear layers to project it to the latent space
        hidden = torch.relu(self.fc(concatenated_attention))  # (batch_size, hidden_dim)
        percent_wrong_pred = torch.sigmoid(self.out(hidden)).squeeze(-1) # (batch_size) - Compute a single distance score per batch

        return percent_wrong_pred, attention_scores
