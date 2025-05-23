import os
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

from transformer.steno_sentences_dataset import StenoSentenceDataset
from transformer.steno_data_collator import StenoDataCollator
from transformer.utils import *
from transformer.config import learning_rate


# Chargement du tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Chargement des datasets
train_dataset = StenoSentenceDataset(TRAIN_WIKI_STENO_CSV, tokenizer)
val_dataset = StenoSentenceDataset(VAL_WIKI_STENO_CSV, tokenizer)
data_collator = StenoDataCollator(tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)

# Chargement du modèle
if os.listdir(TRANSFORMER_DIR):  # Si le dossier n'est pas vide , on charge depuis le dernier checkpoint
    model = T5ForConditionalGeneration.from_pretrained(TRANSFORMER_DIR)
else:
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    print("Entrainement à zero !")
model.to(device)

# Chargement de l'optimiseur
optimizer = AdamW(model.parameters(), lr=learning_rate)
optimizer_state_path = f"{TRANSFORMER_DIR}/optimizer.pt"
if os.path.exists(optimizer_state_path):
    optimizer.load_state_dict(torch.load(optimizer_state_path))

# Chargement du numéro de l'époque
state_path = f"{TRANSFORMER_DIR}/training_state.pt"
if os.path.exists(state_path):
    state = torch.load(state_path)
    start_epoch = state["epoch"]
else:
    start_epoch = 0

# Entraînement avec early stopping depuis le dernier checkpoint
early_stopping(model, tokenizer, optimizer, train_loader, val_loader, start_epoch=start_epoch)

print("Entrainement terminé !")
