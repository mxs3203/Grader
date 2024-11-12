import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assume a single original embedding for the whole SQL code
np.random.seed(42)
original_embedding = np.random.rand(1, 2) * 10  # Random initial 2D embedding for the entire SQL


# Create DataFrame for plotting
sql_tokens = [
    "select", "profilename", "ownerid", "tracktitle", "duration",
    "from", "trackview", "natural", "join", "profile", "natural", "join", "track",
    "where", "(","current_date", "-", "viewStartdatetime",")", "::interval", "<="
]
num_tokens = len(sql_tokens)
ablated_embeddings = original_embedding + np.random.normal(0, 0.1, (num_tokens, 2))  # Small random perturbations

# Calculate distances between the original embedding and each ablated embedding
distances = np.linalg.norm(original_embedding - ablated_embeddings, axis=1)


df_embeddings = pd.DataFrame(ablated_embeddings, columns=['x', 'y'])
df_embeddings['Token'] = sql_tokens
df_embeddings['Type'] = 'Ablated'

# Adding the original embedding to the DataFrame
df_original = pd.DataFrame(original_embedding, columns=['x', 'y'])
df_original['Token'] = 'Original'
df_original['Type'] = 'Original'
df_embeddings = pd.concat([df_original, df_embeddings], ignore_index=True)

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_embeddings, x='x', y='y', hue='Token', style='Type', palette='viridis', s=100)

# Add lines between the original embedding and each ablated version
for i, ((x_orig, y_orig), (x_abl, y_abl)) in enumerate(zip(original_embedding.repeat(num_tokens, axis=0), ablated_embeddings)):
    plt.plot([x_orig, x_abl], [y_orig, y_abl], 'k--', lw=1)
    mid_x, mid_y = (x_orig + x_abl) / 2, (y_orig + y_abl) / 2
    plt.text(mid_x, mid_y, f'{distances[i]:.2f}', color='green', fontsize=8)

plt.title('Impact of Token Ablation on SQL Code Embedding')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(title='Token', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plots/ablation_example.pdf")
