from transformers import PreTrainedTokenizerBase
from typing import List, Dict
import torch

class StenoDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        #print("[DEBUG] Batch received in collator:", batch)
        inputs = [steno_sent["input_text"] for steno_sent in batch]
        targets = [sent["target_text"] for sent in batch]

        model_inputs = self.tokenizer(
            inputs,
            max_length=None,  # padding dynamique
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            targets,
            max_length=None,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        # Remplace les tokens PAD par -100 pour l'entraînement
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        return model_inputs   #  { "input_ids": ..., "attention_mask": ...,  "labels": ... }  (batch, seq_len)
