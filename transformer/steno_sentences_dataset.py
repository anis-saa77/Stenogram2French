# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import pandas as pd


class StenoSentenceDataset(Dataset):
    def __init__(self, csv_file, tokenizer, source_col="steno_sent", target_col="sent", max_source_length=512,
                 max_target_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.source_col = source_col
        self.target_col = target_col
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        steno_sentence = self.data.iloc[idx][self.source_col]
        sentence = self.data.iloc[idx][self.target_col]

        # Format pour T5 : input prefix
        input_text = f"transcrire: {steno_sentence}"
        target_text = sentence

        return {
            "input_text": input_text,
            "target_text": target_text
        }
