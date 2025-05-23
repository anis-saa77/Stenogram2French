# -*- coding: utf-8 -*-
import json
import pandas as pd

from g2p.config import CHAR_TOKENIZER, STENO_CHAR_TOKENIZER, STENOTYPE_TOKENIZER, output_unit


def build_steno_tokenizer(file_path):
    """
        Crée le tokenizer : mapping stenotype → ID
    """
    df = pd.read_csv(file_path)
    all_stenotypes = set()

    for line in df["stenogram"]:
        for token in line.strip().split():
            all_stenotypes.add(token)

    stenotype_to_id = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3,
    }

    for i, steno in enumerate(sorted(all_stenotypes), start=len(stenotype_to_id)):
        stenotype_to_id[steno] = i

    return stenotype_to_id


def build_char_tokenizer(inputs):  # inputs peuvent être des mots français ou des sténogrammes
    """
        Crée le tokenizer : mapping caractère → ID
    """
    vocab = sorted(set("".join(inputs)))  # Le tri est optionnel mais permets d'avoir toujours le même résultat
    char2id = {}
    char2id["<PAD>"] = 0
    char2id["<SOS>"] = 1
    char2id["<EOS>"] = 2
    char2id["<UNK>"] = 3
    for i, char in enumerate(vocab, start=len(char2id)):
        char2id[char] = i

    return char2id


def save_tokenizer(tokenizer, file_path):
    """
        Sauvegarde du tokenizer dans un json
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, ensure_ascii=False, indent=2)


def load_tokenizer(file_path):
    """
        Charge un tokenizer depuis un fichier JSON.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        tokenizer = json.load(f)
    return tokenizer


def invert_tokenizer(tokenizer):
    """
        Crée un dictionnaire inverse : ID → token
    """
    return {int(v): k for k, v in tokenizer.items()}


# Build et sauvegarde des tokenizers    # Déjà fait !
# df = pd.read_csv(STENO_FULL_CSV)
# char2id = build_char_tokenizer(df["word"])
# stenochar2id = build_char_tokenizer(df["stenogram"])
# steno2id = build_steno_tokenizer(STENO_FULL_CSV)

# save_tokenizer(char2id, file_path=CHAR_TOKENIZER) # Déjà fait !
# save_tokenizer(stenochar2id, file_path=STENO_CHAR_TOKENIZER)
# save_tokenizer(steno2id, file_path=STENOTYPE_TOKENIZER)

# Chargement des dictionnaires
# char2id = load_tokenizer(CHAR_TOKENIZER)
# stenochar2id = load_tokenizer(STENO_CHAR_TOKENIZER)
# steno2id = load_tokenizer(STENOTYPE_TOKENIZER)

# Génération des dictionnaires inverses
# id2stenochar = invert_tokenizer(stenochar2id)
# id2steno = invert_tokenizer(steno2id)
