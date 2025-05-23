# -*- coding: utf-8 -*-
import pandas as pd
from tqdm import tqdm
from jiwer import wer
from torch.utils.data import DataLoader

from g2p.config import *
from g2p.utils import beam_search_decode
from g2p.stenodataset import *
from g2p.tokenizer import load_tokenizer, invert_tokenizer


# Chargement des tokenizers
input_tokenizer = load_tokenizer(CHAR_TOKENIZER)  # char2id
target_tokenizer = load_tokenizer(STENOTYPE_TOKENIZER) if output_unit == "stenotype" else load_tokenizer(STENO_CHAR_TOKENIZER)
inverse_tokenizer = invert_tokenizer(target_tokenizer)
pad_idx = input_tokenizer["<PAD>"]

# Chargement du meilleur modèle
g2p_model = torch.load(G2P_MODEL_PATH, weights_only=False)
g2p_model.eval()

# Décodage d’exemples avec beam search
test_df = pd.read_csv(STENO_TEST_CSV)   # Chargement du dataset de test
test_dataset = StenoDataset(test_df, input_tokenizer, target_tokenizer, output_unit=output_unit, max_len=max_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                          collate_fn=lambda x: collate_fn(x, pad_idx))


# Chargement du dataset
test_df = pd.read_csv(STENO_TEST_CSV)
nb_samples = len(test_df)

print("\nÉvaluation sur le jeu de test (WER) :")

total_wer = 0.0

for idx, row in tqdm(test_df.head(nb_samples).iterrows(), total=nb_samples, desc="Évaluation"):
    word = row["word"]
    target = row["stenogram"]

    # Décodage
    predicted_tokens = beam_search_decode(
        g2p_model,
        word,
        input_tokenizer,
        target_tokenizer,
        inverse_tokenizer,
        beam_width=beam_width
    )
    if output_unit == "stenotype":
        predicted_str = " ".join(predicted_tokens)
    else:  # char
        predicted_str = "".join(predicted_tokens)  # On ne met pas d'espace entre chaque caractère
    predicted_str = predicted_str.replace("<UNK>", "").replace("<SOS>", "").replace("<EOS>", "").strip()

    # Calcul du WER entre les deux phrases
    word_wer = wer(target, predicted_str)
    total_wer += word_wer

# Résultat global
average_wer = total_wer / nb_samples
print(f"\nWER : {average_wer:.2%}")