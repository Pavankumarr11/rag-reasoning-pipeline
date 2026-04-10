# training/model.py
# Sets up a small causal language model (distilgpt2) with LoRA adapters.
# LoRA (Low-Rank Adaptation) fine-tunes only a small set of adapter weights,
# keeping the base model frozen — efficient for limited compute.

from transformers import AutoTokenizer, AutoModelForCausalLM


# LoRA configuration values
LORA_RANK       = 8    # Low-rank dimension for adapter matrices
LORA_ALPHA      = 16   # Scaling factor (alpha/rank applied to adapter output)
LORA_DROPOUT    = 0.1  # Dropout applied inside LoRA layers


def load_model_and_tokenizer(model_name: str = "distilgpt2"):
    """
    Load the base model and its tokenizer.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        (model, tokenizer) tuple.
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # distilgpt2 has no pad token by default; reuse EOS as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer


def apply_lora(model):
    """
    Apply LoRA adapters to the model's attention projection layers.

    If the `peft` library is available, we use the standard PEFT LoRA.
    Otherwise we fall back to a minimal manual LoRA simulation that
    injects low-rank adapter matrices into every nn.Linear layer —
    demonstrating the concept without requiring peft.

    Args:
        model: A HuggingFace causal LM.

    Returns:
        Model with LoRA adapters attached (base weights frozen).
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            task_type     = TaskType.CAUSAL_LM,
            r             = LORA_RANK,
            lora_alpha    = LORA_ALPHA,
            lora_dropout  = LORA_DROPOUT,
            bias          = "none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("LoRA applied via PEFT library.")

    except ImportError:
        # ---- Manual LoRA simulation ----
        print("PEFT not found — applying manual LoRA simulation.")
        import torch
        import torch.nn as nn

        class LoRALayer(nn.Module):
            """Wraps an nn.Linear with a low-rank adapter: output += (x @ A @ B) * scale"""
            def __init__(self, original: nn.Linear, rank: int, alpha: float):
                super().__init__()
                self.original = original
                self.rank     = rank
                self.scale    = alpha / rank

                in_f, out_f = original.in_features, original.out_features
                # A is initialised with random Gaussian; B is zero so adapter starts at 0
                self.lora_A = nn.Parameter(torch.randn(in_f,  rank) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(rank,  out_f))

                # Freeze the original weight
                original.weight.requires_grad = False
                if original.bias is not None:
                    original.bias.requires_grad = False

            def forward(self, x):
                base_out  = self.original(x)
                lora_out  = (x @ self.lora_A @ self.lora_B) * self.scale
                return base_out + lora_out

        # Replace every Linear in the model with a LoRALayer
        def _inject_lora(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    setattr(module, name, LoRALayer(child, LORA_RANK, LORA_ALPHA))
                else:
                    _inject_lora(child)

        _inject_lora(model)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model
