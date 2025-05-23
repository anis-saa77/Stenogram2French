import torch

# Path des Datasets
FULL_WIKI_STENO_CSV = "resources/wiki_data/wiki_steno_sentences.csv"
TRAIN_WIKI_STENO_CSV = "resources/t5_datasets/train_wiki_steno_sentences.csv"
VAL_WIKI_STENO_CSV = "resources/t5_datasets/val_wiki_steno_sentences.csv"
TEST_WIKI_STENO_CSV = "resources/t5_datasets/test_wiki_steno_sentences.csv"

# Path du modèle
TRANSFORMER_DIR = "resources/t5_model"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paramètre
num_epochs = 100
patience = 10
learning_rate = 5e-5
beam_width = 5
