# -*- coding: utf-8 -*-
import time
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from g2p.tokenizer import load_tokenizer
from g2p.stenodataset import StenoDataset, collate_fn
from g2p.model import G2P
from g2p.encoder import Encoder
from g2p.decoder import Decoder
from g2p.attention import Attention
from g2p.utils import train_epoch, evaluate
from g2p.config import *

def main():

    # Chargement des tokenizers
    input_tokenizer = load_tokenizer(CHAR_TOKENIZER)  # char2id
    target_tokenizer = load_tokenizer(STENOTYPE_TOKENIZER) if output_unit == "stenotype" else load_tokenizer(STENO_CHAR_TOKENIZER)
    pad_idx = input_tokenizer["<PAD>"]

    # Chargement du modèle
    input_dim = len(input_tokenizer)
    output_dim = len(target_tokenizer)
    encoder_hidden_idm = hidden_dim
    decoder_hidden_idm = hidden_dim

    encoder = Encoder(input_dim, embedding_dim, encoder_hidden_idm).to(device)
    attention = Attention(encoder_hidden_idm, decoder_hidden_idm).to(device)
    decoder = Decoder(output_dim, embedding_dim, encoder_hidden_idm, decoder_hidden_idm, attention).to(device)
    model = G2P(encoder, decoder, device).to(device)

    # Chargement du dataset d'entrainement
    df = pd.read_csv(STENO_TRAIN_CSV)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    train_dataset = StenoDataset(train_df, input_tokenizer, target_tokenizer, output_unit=output_unit, max_len=max_len)
    val_dataset = StenoDataset(val_df, input_tokenizer, target_tokenizer, output_unit=output_unit, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: collate_fn(x, pad_idx))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: collate_fn(x, pad_idx))

    # Optimiseur et fonction de loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Entrainement avec early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        # Entrainement
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        # Validation
        val_loss = evaluate(model, val_loader, criterion)
        elapsed = time.time() - start_time
        print(f"[Époque {epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time : {elapsed:.2f}s")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Sauvegarde du modèle
            torch.save(model, G2P_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping déclenché.")
                break


if __name__ == '__main__':
    main()