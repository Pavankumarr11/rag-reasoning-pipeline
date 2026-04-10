# training/train.py
# Fine-tunes a LoRA-wrapped model on GSM8K using a simple training loop.
# Tokenizes question+answer pairs and trains with causal language modelling loss.

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW

from training.dataset import load_gsm8k, format_sample
from training.model import load_model_and_tokenizer, apply_lora


# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS        = 1          # Keep low for demo; increase for real training
BATCH_SIZE    = 4
LEARNING_RATE = 2e-4
MAX_LENGTH    = 256        # Token sequence length per sample
LOG_EVERY     = 50         # Print loss every N steps
# ─────────────────────────────────────────────────────────────────────────────


class GSM8KTorchDataset(TorchDataset):
    """Wraps the HuggingFace GSM8K slice as a PyTorch Dataset."""

    def __init__(self, hf_dataset, tokenizer):
        self.tokenizer = tokenizer
        self.samples   = [format_sample(s) for s in hf_dataset]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation  = True,
            max_length  = MAX_LENGTH,
            padding     = "max_length",
            return_tensors = "pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # For causal LM, labels == input_ids; pad positions → -100 (ignored in loss)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train(model_name: str = "distilgpt2"):
    """
    Full training pipeline:
      1. Load data & model
      2. Wrap model with LoRA
      3. Run training loop
      4. Save the fine-tuned model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load data
    data           = load_gsm8k(train_size=3000, test_size=1000)
    model, tokenizer = load_model_and_tokenizer(model_name)

    # 2. Apply LoRA
    model = apply_lora(model)
    model = model.to(device)

    # 3. Build DataLoader
    train_dataset = GSM8KTorchDataset(data["train"], tokenizer)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimise only the adapter (LoRA) parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # 4. Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        print(f"\n=== Epoch {epoch + 1} / {EPOCHS} ===")

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % LOG_EVERY == 0:
                avg = total_loss / (step + 1)
                print(f"  Step {step + 1:>4} | Loss: {avg:.4f}")

        epoch_avg = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} complete | Avg Loss: {epoch_avg:.4f}")

    # 5. Save model + tokenizer
    save_path = "gsm8k_lora_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to: {save_path}/")

    return model, tokenizer


if __name__ == "__main__":
    train()
