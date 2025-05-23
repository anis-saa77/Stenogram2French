# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class G2P(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, trg_vocab, max_len=50):
        """
        src: [batch_size, src_len] (séquence de caractères)
        trg: [batch_size, trg_len] (séquence de sténotypes — utilisée seulement pour la taille ici)
        trg_vocab: dictionnaire de tokens ({"<PAD>":0, "<SOS>":1, ...})
        max_len: longueur maximale à prédire
        """

        batch_size = src.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)

        # Encoder
        encoder_outputs, hidden, cell = self.encoder(src)

        start_token = trg_vocab["<SOS>"]
        input_token = torch.LongTensor([start_token] * batch_size).to(self.device)

        for t in range(0, max_len):
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output

            # Token suivant = prédiction actuelle
            top1 = output.argmax(1)
            input_token = top1

        return outputs  # [batch_size, max_len, trg_vocab_size]
