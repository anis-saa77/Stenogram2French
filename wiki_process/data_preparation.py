import re
import pandas as pd

from config import WIKI_CSV, WIKI_CSV_CLEANED

def convert_txt_to_csv(filename, output_file=WIKI_CSV):
    """
        Convertit les fichiers txt en CSV
    """
    # Charger les données
    data = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file.readlines():
            if not line:
                continue  # ignorer les lignes vides

            # Utilise une expression régulière pour séparer l'index du contenu
            match = re.match(r"^(\d+)\s*[^\w\s]?\s*(.*)", line)
            if match:
                index = match.group(1)
                sentence = match.group(2)

            data.append([index, sentence.lower()])  # passer la phrase en minuscule

    # Convertir en DataFrame Pandas
    df = pd.DataFrame(data, columns=["id", "sentences"])

    # Sauvegarder en CSV
    df.to_csv(output_file, index=False, encoding="utf-8")

def is_clean(text):
    """
        Regex pour filtrer les phrases contenant uniquement lettres, espaces, ponctuation classique
        Exclut si chiffre ou caractère spécial
    """
    return not re.search(r"[\d@#~^¤$€%=*\\_{}\[\]|<>/`]", text)

def clean_data(filename=WIKI_CSV, output_file=WIKI_CSV_CLEANED):
    """
        Nettoie les données dans WIKI_CSV afin de filtrer les phrases contenant des chiffres ou des caractères spéciaux
    """
    df = pd.read_csv(filename)

    # Appliquer le filtre
    df_cleaned = df[df['sentences'].apply(is_clean)]

    # Sauvegarde dans un nouveau fichier
    df_cleaned.to_csv(output_file, index=False)

# Création du fichier CSV
#convert_txt_to_csv(WIKI_TXT, WIKI_CSV)  # Fichier texte supprimé

# Nettoyage des données
clean_data(WIKI_CSV, WIKI_CSV_CLEANED)