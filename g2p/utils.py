# -*- coding: utf-8 -*-
from tqdm import tqdm

from g2p.tokenizer import load_tokenizer
from g2p.config import *


# Tokenizers
input_tokenizer = load_tokenizer(CHAR_TOKENIZER)  # char2id
target_tokenizer = load_tokenizer(STENOTYPE_TOKENIZER) if output_unit == "stenotype" else load_tokenizer(STENO_CHAR_TOKENIZER)
pad_idx = input_tokenizer["<PAD>"]
sos_idx = input_tokenizer["<SOS>"]
eos_idx = input_tokenizer["<EOS>"]
unk_idx = input_tokenizer["<UNK>"]


def sequence_loss(predictions, targets, criterion):
    """
    predictions: [batch_size, seq_len, vocab_size] — logits
    targets: [batch_size, seq_len] — vrais indices
    pad_idx: index du token <PAD> à ignorer
    """

    # Aplatir tout sauf vocab_size
    predictions = predictions.view(-1, predictions.size(-1))  # [batch_size * seq_len, vocab_size]
    targets = targets.view(-1)  # [batch_size * seq_len]

    loss = criterion(predictions, targets)

    return loss


def train_epoch(model, dataloader, optimizer, criterion):
    """ Entrainement sur une époque """

    model.train()
    total_loss = 0

    for src, trg in tqdm(dataloader, desc="Entrainement"):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg, trg_vocab=target_tokenizer, max_len=trg.shape[1])  # (batch_size, trg_len, vocab_size)

        # Calcul de la loss
        loss = sequence_loss(output, trg, criterion)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, loader, criterion):
    """
    Fonction d'évaluation du modèle sur le jeu de validation.
    """
    model.eval()  # Met le modèle en mode évaluation
    total_loss = 0
    with torch.no_grad():  # Désactive le calcul des gradients pendant l'évaluation
        for src, trg in tqdm(loader, desc="Évaluation"):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, trg_vocab=target_tokenizer, max_len=trg.shape[1])  # [batch_size, seq_len, output_dim]
            # Calcul de la loss
            loss = sequence_loss(output, trg, criterion)
            total_loss += loss.item()

    return total_loss / len(loader)


def beam_search_decode(model, word, input_tokenizer, target_tokenizer, inverse_tokenizer, beam_width):
    """ Décodage avec beam search """

    model.eval()
    with torch.no_grad():
        token2id = target_tokenizer   # Tokenizer des sorties (steno2id ou stenochar2id)
        id2token = inverse_tokenizer  # Inverse du tokenizer des sorties

        # Encode l'entrée
        input_ids = [input_tokenizer.get(c, unk_idx) for c in word]
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # batch=1

        encoder_outputs, hidden, cell = model.encoder(input_tensor)  # on applique l'encodage

        sequences = [[["<SOS>"], 0.0, hidden, cell]]  # (tokens, score, hidden, cell)
        for i in range(max_len):  # taille max d'une sortie sténogramme
            all_candidates = []
            for seq, score, h, c in sequences:
                # Si la séquence est entièrement générée on l'ignore
                if seq[-1] == id2token[eos_idx]:
                    all_candidates.append((seq, score, h, c))
                    continue

                token_id = token2id.get(seq[-1], unk_idx)
                last_token = torch.tensor([token_id], dtype=torch.long, device=device)
                output, h, c, _ = model.decoder(last_token, h, c, encoder_outputs)  # Passer h et c au décodeur

                # On récupère les prochains tokens les plus probables
                probs = torch.log_softmax(output, dim=1)
                topk = torch.topk(probs, beam_width)

                for j in range(beam_width):
                    token_id = topk.indices[0][j].item()
                    token_score = topk.values[0][j].item()
                    token = id2token.get(token_id, "<UNK>")
                    all_candidates.append((seq + [token], score + token_score, h, c))

            # Tri en fonction du score de la séquence
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            if all(seq[-1] == "<EOS>" for seq, _, _, _ in sequences):
                break
        return sequences[0][0]  # on retourne la meilleure séquence


