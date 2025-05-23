import pandas as pd

from config import *


def convert_txt_to_csv(filename, output_file):
    """
        Convertit les fichiers txt en CSV
    """
    # Charger les données
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" :: ")
            if len(parts) == 2:
                data.append(parts)

    # Convertir en DataFrame Pandas
    df = pd.DataFrame(data, columns=["word", "stenogram"])

    # Sauvegarder en CSV
    df.to_csv(output_file, index=False, encoding="utf-8")

# Convertir les txt en csv
# convert_txt_to_csv(STENO_FULL_TXT, STENO_FULL_CSV)   # Fichier txt supprimé
# convert_txt_to_csv(STENO_TRAIN_TXT, STENO_TRAIN_CSV) # Fichier txt supprimé
# convert_txt_to_csv(STENO_TEST_TXT, STENO_TEST_CSV)   # Fichier txt supprimé

# Taille max des sténogramme (en caractère)
df = pd.read_csv(STENO_FULL_CSV)
max_char_len = df["stenogram"].apply(len).max()
print(f"Longueur maximale (caractères) : {max_char_len}")

# Taille max des sténogramme (en sténotypes)
max_word_len = df["stenogram"].apply(lambda x: len(str(x).split())).max()
print(f"Longueur maximale (mots) : {max_word_len}")
