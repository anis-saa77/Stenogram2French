## Structure du projet
````
+---g2p             # Fichiers relatifs au modèle g2p
    >---config.py           # Configuration des hparams, paths et unité de sortie (stenotype/caractère)
    >---tokenizer.py        # Fonctions de création, sauvegarde et chargement des dictionnaires
    >---stenodataset.py     # Classe custom du dataset
    >---encoder             # Classe du modèle encodeur
    >---attention           # Classe du mécanisme d'attention
    >---decoder             # Classe du modèle decoder
    >---model               # Classe du modèle G2P
    >---utils               # Fonctions utilitaire (train, eval, etc.)
    
+---wiki_process    # Fichiers relatifs à la génération du corpus
    >---config.py              # Configuration (Chemins)
    >---data_preparation.py    # Traitement des données wikipedia
    >---corpus_generation.py   # Fonction de génération du corpus
    
+---transformer     # Fichiers relatifs au modèle t5
    >---config.py                   # Configuration (chemins et hparam)
    >---data_prep.py                # Split du corpus en 3 datasets (train, val, test)
    >---steno_sentences_dataset.py  # Classe custom du dataset
    >---steno_data_collator.py      # Classe du collator
    >---utils.py                    # Fonctions utilitaires (train, eval, etc.)

+---resources       # Ressources utilisées dans le projet (modèles et datasets)

|---train_g2p.py    # Script d'entrainement du modèle G2P
|---test_g2p.py     # Script d'évaluation du modèle G2P
|---corpus_gen.py   # Script de génération du corpus (phrase sténogramme | phrase fr)
|---train_t5.py     # Script d'entrainement du modèle transformer t5
|---test_t5.py      # Script d'évaluation du modèle transformer t5

|---train_g2p.sh
|---test_g2p.sh
|---corpus_gen.sh
|---train_t5.sh
|---test_t5.sh

|---requirements.txt # Fichier requirements (Compatible windows) 
|---README.md        # C'est ici !
````

## Modèle G2P

### Configuration :

Dans `g2p/config.py` configurez les différents hyperparamètres ainsi que **output_unit** (
l'unité de sortie) pour entrainer/tester un modèle qui gènére soit des sténotypes, soit des caractères.

### Execution :

**Entrainement :**
````commandline
python train_g2p.py
````
**ou sur joyeux :**
````commandline
sbatch train_g2p.sh
````

**Test :**
````commandline
python test_g2p.py
````
**sur joyeux :**
````commandline
sbatch test_g2p.sh
````

### Résultats du G2P:
#### Modèle G2P générant des sténotypes :
- **WER : 19.08%**

#### Modèle G2P générant des caractères :
- **WER : 17.01%**

## Génération du corpus
### Donnée Wikipedia :
Les phrases de wikipédia en été importé depuis : https://wortschatz.uni-leipzig.de/en/download/French \
Le corpus utilisé est celui de 2021 avec 100 000 phrases.\
Il a ensuite été filtré, en supprimant les phrases avec des caractères spéciaux et/ou des chiffres.\
Seules les phrases avec une ponctuation ordinaire sont gardées. (76 000 phrases)

### Génération du coprus :
Le corpus est généré avec le modèle g2p basé sur la génération caractère par caractère,\
en utilisant la recherche par faisceau (beam search) pour améliorer les résultats.\
Chemin du corpus : `resources/wiki_data/wiki_steno_sentences.csv`

### Execution :

````commandline
python corpus_gen.py
````
**ou sur joyeux :**
````commandline
sbatch corpus_gen.sh
````

## Fine-tuning du modèle t5-small

### Répartition du corpus : 
- Donnée de test : 5000 phrases - `resources/t5_datasets/test_wiki_steno_sentences.csv`
- Donnée d'entrainement : 90% restants soit ~64000 phrases - `resources/t5_datasets/train_wiki_steno_sentences.csv`
- Donnée de validation : 10% restants soit ~7000 phrases - `resources/t5_datasets/val_wiki_steno_sentences.csv`

### Execution :

**Entrainement :**
````commandline
python train_t5.py
````
**sur joyeux :**
````commandline
sbatch train_t5.sh
````

**Test :**
````commandline
python test_t5.py
````
**sur joyeux :**
````commandline
sbatch test_t5.sh
````

### Télécharger le modèle fine-tuné
Le modèle finetuné n'a pas pu être inclus dans le dépôt en raison de sa taille (>100Mo).  
Si vous souhaitez le tester, veuillez le télécharger ici: [Lien Google Drive](https://drive.google.com/file/d/1jxG3TR8j5GNaihyme6GqOPzcwRiCgTly/view?usp=sharing).
Attention, il fait **639Mo !** (~240Mo pour le modèle + ~400 pour l'optimiseur)\
Une fois téléchargé, décompressez le zip dans le dossier suivant : `resources/t5_model/`

### Fichiers de logs
Le suivi des entrainements et des évaluations peut être consultés dans les fichiers
`logs/`.

## Meilleur résultat :
**Avec la configuration suivante :**
- Patience de l'early stopping : 3
- Taille du faisceau : 5 (Beam search)
- Pas d'apprentissage : 0.00005

L'entrainement s'est arrêté après 55 époques. Cependant nous avons repris l'entrainement
depuis ce dernier checkpoint avec cette fois **une patience de 5** pour trouver une meilleure solution.

**A la 64ème époque**, on obtient : **WER = 14,01%** sur l'ensemble de test (86% des mots sont bien traduit).
