import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader

from transformer.config import TEST_WIKI_STENO_CSV, TRANSFORMER_DIR
from transformer.steno_sentences_dataset import StenoSentenceDataset
from transformer.steno_data_collator import StenoDataCollator
from transformer.utils import evaluate

# Passage sur GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle et du tokenizer
model_dir = TRANSFORMER_DIR  # dossier du modèle entraîné
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_dir)


def predict(input, max_length=128):
    """ Fonction pour générer une prédiction à partir d'une entrée """
    input_text = f"transcrire: {input}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True
    ).to(device)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


def predict_single_example(steno_input):
    """ Tester la prédiction sur une seule entrée """
    output = predict(steno_input)
    print(f"\nTranscription : {output}\n")


if __name__ == "__main__":
    # Test des sorties générées
    #predict_single_example("POULRULKO# SEL F*ELAALYU$ TE$D ELI#$ TRU (A#L$TUDDE$DPLUYL Y( NLA# T*ROIP*E( EE(EL$S  SEI# SA# SID(MA#YU$( OLAALKU LA$")
    #predict_single_example("MI$A  # SKENNU K*EUSKAL P*O NO")

    # Chargement du dataset
    test_dataset = StenoSentenceDataset(TEST_WIKI_STENO_CSV, tokenizer)
    data_collator = StenoDataCollator(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

    # Evaluation sur l'ensemble de test (5000 phrases)
    evaluate(model, tokenizer, test_loader)
