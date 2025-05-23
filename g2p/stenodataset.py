# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class StenoDataset(Dataset):
    def __init__(self, dataframe, input_tokenizer, target_tokenizer, output_unit, max_len=50):
        """
        output_unit: soit 'char' pour caractère par caractère, soit 'stenotype' pour sténotype par unité
        """
        self.data = dataframe
        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer
        self.output_unit = output_unit
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.data.iloc[idx]["word"]
        steno = self.data.iloc[idx]["stenogram"]

        # Encode input word caractère par caractère
        x = [self.input_tokenizer.get(c, self.input_tokenizer["<UNK>"]) for c in word.lower()]

        # Encode target soit en sténotypes soit en caractères
        if self.output_unit == "stenotype":
            trg_units = steno.split()
        elif self.output_unit == "char":
            #trg_units = list(steno.replace(" ", ""))  # fusionne puis sépare en caractères (sans les espaces)
            trg_units = list(steno)  # Convertit la chaine en liste de caractères (espaces compris)

        # Ajout des SOS et EOS dans les sorties
        trg = []
        trg.append(self.target_tokenizer["<SOS>"])  # Ajoute <SOS> à la liste
        trg.extend([self.target_tokenizer.get(tok, self.target_tokenizer.get("<UNK>")) for tok in
                    trg_units])  # Ajoute les autres ID
        trg.append(self.target_tokenizer["<EOS>"])  # Ajoute <EOS> à la fin

        return torch.tensor(x, dtype=torch.long), torch.tensor(trg, dtype=torch.long)

def collate_fn(batch, pad_token):
    """
    Padding dynamique pour un batch.
    """
    srcs, trgs = zip(*batch)
    srcs_padded = pad_sequence(srcs, batch_first=True, padding_value=pad_token)
    trgs_padded = pad_sequence(trgs, batch_first=True, padding_value=pad_token)
    return srcs_padded, trgs_padded
