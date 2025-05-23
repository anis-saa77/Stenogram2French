from tqdm import tqdm
from jiwer import wer

from transformer.config import *


def early_stopping(model, tokenizer, optimizer, train_loader, val_loader, start_epoch):
    """ Early stopping avec sauvegarde du mmodèle et l'état de l'optimiseur comme dernier checkpoint"""

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, train_loader, optimizer)
        if epoch < 30:  # Inutil de faire l'early stopping avant les 30 époques
            continue

        val_loss, val_wer = evaluate(model, tokenizer, val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | WER: {val_wer*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Sauvegarde modèle et optimizer
            model.save_pretrained(TRANSFORMER_DIR)
            tokenizer.save_pretrained(TRANSFORMER_DIR)
            torch.save(optimizer.state_dict(), f"{TRANSFORMER_DIR}/optimizer.pt")
            torch.save({"epoch": epoch+1}, f"{TRANSFORMER_DIR}/training_state.pt")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered !")
                break


def train(model, train_loader, optimizer):
    """ Entrainement """
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss


def evaluate(model, tokenizer, val_loader):
    """ Evaluation des prédictions avec calcul du WER """
    model.eval()
    val_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):
            batch = {k: v.to(device) for k, v in batch.items()}  # passage sur gpu
            labels = batch["labels"]
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()

            # Génération + évaluation #######  Trop lent !!!!!!!!!!!!!!!!!
            # generated_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], max_length=512)
            # decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # labels[labels == -100] = tokenizer.pad_token_id  # Remplacer -100 par pad_token_id
            # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Prédictions rapides par argmax sur logits
            # logits = outputs.logits
            # pred_ids = logits.argmax(dim=-1)

            # Génération des séquences
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
                num_beams=beam_width,
                early_stopping=True
            )

            # Préparation pour le décodage
            labels = labels.clone()
            labels[labels == -100] = tokenizer.pad_token_id

            # Décodage
            decoded_preds = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels.cpu(), skip_special_tokens=True)

            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)

    # Nettoyage des chaînes
    decoded_preds = [pred.strip().lower() for pred in all_preds]
    decoded_labels = [label.strip().lower() for label in all_labels]

    # Calcul du WER
    wer_score = wer(decoded_labels, decoded_preds)
    avg_val_loss = val_loss / len(val_loader)

    return avg_val_loss, wer_score
