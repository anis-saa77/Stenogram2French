import pandas as pd
from sklearn.model_selection import train_test_split

from transformer.config import *
def split_dataset(full_dataset_path=FULL_WIKI_STENO_CSV, test_size=5000, val_ratio=0.1, seed=42):
    """
    Divise le dataset initial en trois fichiers :
    - test.csv (5k échantillons)
    - train.csv (~64k / 90%)
    - val.csv (~7k / 10%)

    full_dataset_path: Chemin du fichier CSV source (avec colonnes `steno_sent` et `sent`)
    test_size: Nombre de lignes dans le fichier test
    val_ratio: Pourcentage de validation dans les données restantes
    """

    # Chargement du dataset complet
    df = pd.read_csv(full_dataset_path)

    # Séparation test / reste
    df_test = df.sample(n=test_size, random_state=seed)
    df_remaining = df.drop(df_test.index)

    # Séparation train / val à partir du reste
    df_train, df_val = train_test_split(
        df_remaining,
        test_size=val_ratio,
        random_state=seed
    )

    # Sauvegarde
    df_train.to_csv(TRAIN_WIKI_STENO_CSV, index=False)
    df_val.to_csv(VAL_WIKI_STENO_CSV, index=False)
    df_test.to_csv(TEST_WIKI_STENO_CSV, index=False)

# Création des datasets
split_dataset(full_dataset_path=FULL_WIKI_STENO_CSV, test_size=5000, val_ratio=0.1, seed=42)