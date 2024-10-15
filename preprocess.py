import re
from collections import Counter
import pickle as pk
import pandas as pd

"""
  Tokenizes a given SQL query string.

  Steps:
  1. Replaces Windows-style line breaks (`\r\n`) with Unix-style (`\n`) and replaces tab characters with spaces.
  2. Converts the SQL query to lowercase to ensure consistent tokenization regardless of case.
  3. Uses a regular expression to extract tokens:
     - `[\w']+` matches words, column names, and values (alphanumeric and underscores).
     - `[.,;()=><!~+*/%-]` matches common SQL symbols and operators.

  Returns:
      A list of tokens extracted from the SQL query.
  """
def sql_tokenizer(sql_code):
    sql_code = re.sub(r'--.*?(\r\n|\n)', ' ', sql_code)  # Single-line comments
    sql_code = re.sub(r'/\*.*?\*/', ' ', sql_code, flags=re.DOTALL) # multiline comments
    sql_code = sql_code.replace('\r\n', ' ').replace('\t', ' ').lower()
    sql_code = re.sub(r'\s+', ' ', sql_code).strip()
    pattern = r"[\w']+|[.,;()=><!~+*/%-]"
    tokens = re.findall(pattern, sql_code)

    return tokens

"""
    Pads or truncates a sequence to a specified maximum length.

    Args:
        sequence (list): A list of tokens or indices representing a tokenized SQL query.
        max_length (int): The length to which the sequence should be padded or truncated.
        pad_value (int): The value to use for padding the sequence (default is 0, which represents the <PAD> token).

    Returns:
        A list where the original sequence is either:
        - Padded with `pad_value` if it is shorter than `max_length`.
        - Truncated if it is longer than `max_length`.
"""
def pad_sequence(sequence, max_length, pad_value=0):
    # If the sequence is shorter than max_length, pad with pad_value
    if len(sequence) < max_length:
        return sequence + [pad_value] * (max_length - len(sequence))
    # If the sequence is longer than max_length, truncate it
    else:
        return sequence[:max_length]


"""
    Converts a list of tokens into their corresponding indices using a vocabulary.

    Args:
        tokens (list): A list of tokens (words or symbols) from a tokenized SQL query.
        vocab (dict): A dictionary mapping each unique token to a specific integer index.

    Returns:
        A list of integers where each token is replaced by its corresponding index from the `vocab`.
        If a token is not found in the `vocab`, it is replaced by the index of the <UNK> (unknown) token.
"""
def tokens_to_indices(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

data = pd.read_csv("data/data.csv")
data = data[data['maxpoints'] != 0]
data['percentWrong'] = 1-(data['teachergradedpoints']/data['maxpoints'])
all_text = ' '.join(data['correctsolution'].tolist())
tokens = sql_tokenizer(all_text)
token_counter = Counter(tokens)
vocab = {token: idx for idx, (token, _) in enumerate(token_counter.items(), start=1)}

# Optionally, add special tokens like <PAD> and <UNK>
vocab['<PAD>'] = 0
vocab['<UNK>'] = len(vocab) + 1
print(len(vocab))
pk.dump(vocab, open("data/vocab.pkl", "wb"))
data['correctsolution_indices'] = data['correctsolution'].apply(lambda sql: tokens_to_indices(sql_tokenizer(sql), vocab))
data['studentsolution_indices'] = data['studentsolution'].apply(lambda sql: tokens_to_indices(sql_tokenizer(sql), vocab))
max_length = data['studentsolution_indices'].apply(len).max()
data['correctsolution_padded'] = data['correctsolution_indices'].apply(lambda seq: pad_sequence(seq, max_length))
data['studentsolution_padded'] = data['studentsolution_indices'].apply(lambda seq: pad_sequence(seq, max_length))

data[['studentsolution_padded', 'correctsolution_padded', 'percentWrong']].to_pickle("data/processed_data.pkl")