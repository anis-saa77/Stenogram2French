import torch
import pandas as pd
from tqdm import tqdm

from wiki_process.config import WIKI_CSV_CLEANED, WIKI_STENO_CORPUS
from g2p.config import CHAR_TOKENIZER, STENOTYPE_TOKENIZER, STENO_CHAR_TOKENIZER, beam_width, output_unit
from g2p.utils import beam_search_decode
from g2p.tokenizer import load_tokenizer, invert_tokenizer

def steno_corpus_generation(model, corpus_size, filepath=WIKI_CSV_CLEANED, output_file=WIKI_STENO_CORPUS):
    """ Création du corpus stenogram :: sentence ""à l'aide du modèle G2P """
    corpus = []

    # Chargement des tokenizers
    input_tokenizer = load_tokenizer(CHAR_TOKENIZER)  # char2id
    target_tokenizer = load_tokenizer(STENOTYPE_TOKENIZER) if output_unit == "stenotype" else load_tokenizer(
        STENO_CHAR_TOKENIZER)
    inverse_tokenizer = invert_tokenizer(target_tokenizer)

    # Chargement des données
    df = pd.read_csv(filepath)

    # Génération du corpus
    for index, row in tqdm(df.head(corpus_size).iterrows(), total=corpus_size, desc="Génération du corpus"):
        sentence = row['sentences'].strip('"')  # Retirer les guillemets dans les extémitées
        words = sentence.split()
        steno_sent = ""

        # Conversion de chaque mot de la phrase en sténogramme
        for word in words:
            predicted_tokens = beam_search_decode(model, word, input_tokenizer, target_tokenizer, inverse_tokenizer, beam_width=beam_width)
            if output_unit == "stenotype":
                predicted_stenogram = ' '.join(predicted_tokens)
            elif output_unit == "char":
                predicted_stenogram = ''.join(predicted_tokens)
            else:
                raise ValueError("Le paramètre output_unit prend les valeurs 'stenotype' ou 'char' !")
            #print("word : ", word)
            #print("predicted_stenogram : ", predicted_stenogram)
            predicted_stenogram = predicted_stenogram.replace("<SOS>", "").replace("<EOS>", "").strip() # Supprime SOS et EOS

            # Ajout du moté généré à la phrase
            steno_sent += predicted_stenogram

        corpus.append([steno_sent, sentence])

    # Conversion en DataFrame Pandas
    df = pd.DataFrame(corpus, columns=["steno_sent", "sent"])

    # Sauvegarde en CSV
    df.to_csv(output_file, index=False, encoding="utf-8")
