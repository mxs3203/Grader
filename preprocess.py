import re
from collections import Counter
import pickle as pk
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Predefined list of SQL keywords and operators
SQL_KEYWORDS = {
    'select', 'from', 'where', 'insert', 'update', 'delete', 'join', 'inner', 'left', 'right', 'full', 'outer',
    'on', 'group', 'by', 'order', 'having', 'limit', 'distinct', 'count', 'sum', 'avg', 'min', 'max',
    'and', 'or', 'not', 'is', 'null', 'as', 'like', 'in', 'between', 'case', 'when', 'then', 'else', 'end',
    'create', 'table', 'primary', 'key', 'foreign', 'references', 'drop', 'alter', 'add', 'modify',
    'union', 'intersect', 'except', 'all', 'exists', 'truncate', 'values'
}

# SQL operators and symbols to keep
SQL_OPERATORS = {'.', ',', ';', '(', ')', '=', '>', '<', '+', '-', '*', '/', '%', '!', '~'}

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
    pattern = r"[\w]+|[.,;()=><!~+*/%-]|'"
    tokens = re.findall(pattern, sql_code)
    processed_tokens = []
    for token in tokens:
        if '.' != token:
            processed_tokens.append(token)
        else:
            print("We dont want this: ",token)

    #filtered_tokens = [token for token in tokens if token in SQL_KEYWORDS or token in SQL_OPERATORS]

    return processed_tokens#,filtered_tokens

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
def tokens_to_indices(tokens, vocab, is_student=False):
    if is_student:
        # Map student tokens to <UNK> if they are not in the correct vocab
        return [vocab.get(token, vocab['<UNK>']) for token in tokens]
    else:
        # Map correct tokens directly
        return [vocab[token] for token in tokens]

data = pd.read_csv("data/data.csv")
data = data[data['maxpoints'] != 0]
#data = data.sample(1000)

data['percent'] = data['teachergradedpoints']/data['maxpoints']
data['percentWrong'] = 1-(data['teachergradedpoints']/data['maxpoints'])
'''
    Make one string with everything to count the words and make vocabulary
'''
# Tokenize all SQL queries and build vocabulary after tokenization
data['correctsolution_tokenized'] = data['correctsolution'].apply(sql_tokenizer)
data['studentsolution_tokenized'] = data['studentsolution'].apply(sql_tokenizer)
correct_tokens_all = [token for tokens in data['correctsolution_tokenized'] for token in tokens]
correct_token_counter = Counter(correct_tokens_all)
vocab = {token: idx for idx, token in enumerate(correct_token_counter, start=1)}
vocab['<PAD>'] = 0
vocab['<UNK>'] = len(vocab) + 1
pk.dump(vocab, open("data/vocab.pkl", "wb"))
# Map correct solution tokens to indices
data['correct_indices'] = data['correctsolution_tokenized'].apply(lambda tokens: tokens_to_indices(tokens, vocab))
# Map student solution tokens to indices, with <UNK> for unknown tokens
data['student_indices'] = data['studentsolution_tokenized'].apply(lambda tokens: tokens_to_indices(tokens, vocab, is_student=True))

'''
    Preprocess the data and save it to a file
'''
MAX_SEQUENCE_LENGTH = 180
data['studentsolution_len'] = data['student_indices'].apply(len)
data['correctsolution_len'] = data['correct_indices'].apply(len)
plt.hist(data['studentsolution_len'],bins=40)
plt.hist(data['correctsolution_len'], bins=40)
plt.legend(['studentsolution', 'correctsolution'])
plt.show()
print(np.shape(data))
data = data[(data['correctsolution_len'] <= MAX_SEQUENCE_LENGTH) & (data['studentsolution_len'] <= MAX_SEQUENCE_LENGTH)]
print(np.shape(data))
data['correctsolution_padded'] = data['correct_indices'].apply(lambda seq: pad_sequence(seq, MAX_SEQUENCE_LENGTH ))
data['studentsolution_padded'] = data['student_indices'].apply(lambda seq: pad_sequence(seq, MAX_SEQUENCE_LENGTH))

data[['studentsolution_padded', 'correctsolution_padded', 'percentWrong', 'percent']].to_pickle("data/processed_data.pkl")
data[['studentsolution_padded', 'correctsolution_padded', 'percentWrong', 'percent']].to_csv("data/processed_data.csv")

