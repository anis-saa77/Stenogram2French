import torch
import os

# ---------- PARAMETRES ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
embedding_dim = 128
hidden_dim = 256
num_epochs = 100
patience = 5  # patience de l'early-stopping
learning_rate = 0.001
beam_width = 5
output_unit = "char"  # output_unit = "stenotype"  ou "char"
# longueur max (en sténotype ou caractères) de la sortie
max_len = 10 if output_unit == "stenotype" else 50  # valeurs estimées dans g2p/data_prep.py

# ----------- CHEMINS -------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Modèle
if output_unit == "stenotype":
    G2P_MODEL_PATH = os.path.join(BASE_DIR, "../resources/g2p_model/g2p_model_with_steno_tokens.pt")
else:  # output_unit == "char"
    G2P_MODEL_PATH = os.path.join(BASE_DIR, "../resources/g2p_model/g2p_model_with_chars_tokens.pt")

# Fichiers txt (Supprimés après conversion en csv)
# STENO_FULL_TXT = os.path.join(BASE_DIR, "../resources/txt/full.steno.txt")
# STENO_TRAIN_TXT = os.path.join(BASE_DIR, "../resources/txt/train+valid.steno.txt")
# STENO_TEST_TXT = os.path.join(BASE_DIR, "../resources/txt/test.steno.txt")

# Fichiers CSV
STENO_FULL_CSV = os.path.join(BASE_DIR, "../resources/g2p_datasets/full.steno.csv")
STENO_TRAIN_CSV = os.path.join(BASE_DIR, "../resources/g2p_datasets/steno_train_dataset.csv")
STENO_TEST_CSV = os.path.join(BASE_DIR, "../resources/g2p_datasets/steno_test_dataset.csv")

# Tokenizers JSON
CHAR_TOKENIZER = os.path.join(BASE_DIR, "../resources/g2p_tokenizers/char_tokenizer.json")             # Dictionnaire des caractères des mots rançais
STENOTYPE_TOKENIZER = os.path.join(BASE_DIR, "../resources/g2p_tokenizers/stenotype_tokenizer.json")   # Dictionnaire des sténotypes
STENO_CHAR_TOKENIZER = os.path.join(BASE_DIR, "../resources/g2p_tokenizers/steno_char_tokenizer.json") # Dictionnaire des caractères des sténogrammes
