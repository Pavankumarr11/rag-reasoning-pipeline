# training/evaluate.py
# Evaluates the fine-tuned model on GSM8K test samples.
# Metric: Exact Match — the predicted final answer must match the ground-truth answer exactly.

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from training.dataset import load_gsm8k


MAX_NEW_TOKENS = 128   # Max tokens to generate for each answer
EVAL_BATCH    = 50     # Number of test samples to evaluate (set lower to run quickly)


def extract_final_number(text: str) -> str:
    """
    GSM8K answers end with '#### <number>'.
    Extract and return that number as a string.
    If the pattern is absent, return the last digit sequence found.
    """
    # Try the canonical GSM8K format first
    match = re.search(r'####\s*([\d,]+)', text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fall back: last number in the text
    numbers = re.findall(r'\d+', text)
    return numbers[-1] if numbers else ""


def evaluate(model_path: str = "gsm8k_lora_model"):
    """
    Generate answers for test samples and compute exact-match accuracy.

    Args:
        model_path: Path to the saved fine-tuned model directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {model_path}  (device: {device})")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    # Load test split
    data      = load_gsm8k(test_size=EVAL_BATCH)
    test_data = data["test"]

    correct = 0
    total   = len(test_data)

    print(f"\nEvaluating on {total} samples...\n")

    for i, sample in enumerate(test_data):
        question    = sample["question"].strip()
        true_answer = extract_final_number(sample["answer"])

        # Build a prompt (question only; model must predict the answer)
        prompt = f"Question: {question}\nAnswer:"

        inputs = tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 200,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens = MAX_NEW_TOKENS,
                pad_token_id   = tokenizer.eos_token_id,
                do_sample      = False,   # greedy decoding for determinism
            )

        # Decode only the newly generated tokens
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        predicted_answer = extract_final_number(generated)

        is_correct = predicted_answer == true_answer
        if is_correct:
            correct += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}]  True: {true_answer:>6}  |  Predicted: {predicted_answer:>6}  |  {'✓' if is_correct else '✗'}")

    accuracy = correct / total * 100
    print(f"\n{'='*40}")
    print(f"Exact Match Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"{'='*40}")
    return accuracy


if __name__ == "__main__":
    evaluate()
