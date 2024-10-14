import ast

import torch
from torch.utils.data import Dataset
import pandas as pd


class QuestionDataset(Dataset):
    def __init__(self, data, column_names, device):
        self.data = data
        self.correct_column = column_names[1]
        self.student_column = column_names[0]
        self.points_column = column_names[2]
        self.device = device

    def __len__(self):
        """Return the number of rows in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        correct_sql = self.data.iloc[idx][self.correct_column]
        student_sql = self.data.iloc[idx][self.student_column]
        if isinstance(correct_sql, str):
            correct_sql = ast.literal_eval(correct_sql)
        if isinstance(student_sql, str):
            student_sql = ast.literal_eval(student_sql)
        correct_sql_tensor = torch.tensor(correct_sql, dtype=torch.long, device=self.device)
        student_sql_tensor = torch.tensor(student_sql, dtype=torch.long, device=self.device)
        percent = torch.tensor(self.data.iloc[idx][self.points_column], dtype=torch.float32,
                               device=self.device)
        return {
            'correct_sql': correct_sql_tensor,
            'student_sql': student_sql_tensor,
            'percentWrong': percent
        }
