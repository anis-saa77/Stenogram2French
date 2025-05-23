import os

# Chemins
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

#G2P_MODEL_PATH = os.path.join(BASE_DIR, "../resources/g2p_model/g2p_model_with_steno_tokens.pt")  # Génération de sténotypes
G2P_MODEL_PATH = os.path.join(BASE_DIR, "../resources/g2p_model/g2p_model_with_chars_tokens.pt")   # Génératioin de caractères

#WIKI_TXT = os.path.join(BASE_DIR, "../resources/wiki_data/fra_wikipedia_2021_100K-sentences.txt")  # Fichier supprimé
WIKI_CSV = os.path.join(BASE_DIR, "../resources/wiki_data/fra_wikipedia_2021_100K-sentences.csv")   # Data Wikipedia
WIKI_CSV_CLEANED = os.path.join(BASE_DIR, "../resources/wiki_data/fra_wikipedia_2021_100K-sentences_cleaned.csv") # Data Wikipedia Filtré
WIKI_STENO_CORPUS = os.path.join(BASE_DIR, "../resources/wiki_data/wiki_steno_sentences.csv")   # Corpus généré
