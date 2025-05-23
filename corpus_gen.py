import torch
import pandas as pd

from g2p.config import device
from wiki_process.config import WIKI_CSV_CLEANED, WIKI_STENO_CORPUS, G2P_MODEL_PATH
from wiki_process.corpus_generation import steno_corpus_generation


""" Génération du corpus d'apprentissage (steno_sentence,sentence) """

# Chargement du modèle
g2p_model = torch.load(G2P_MODEL_PATH)
g2p_model = g2p_model.to(device)
g2p_model.eval()

# Taille du corpus
df = pd.read_csv(WIKI_CSV_CLEANED)
CORPUS_SIZE = len(df)  # Environ 75000 après filtrage
#CORPUS_SIZE = 10      # Pour tester sur quelques phrases

# Génération du corpus
steno_corpus_generation(g2p_model, corpus_size=CORPUS_SIZE, filepath=WIKI_CSV_CLEANED, output_file=WIKI_STENO_CORPUS)
