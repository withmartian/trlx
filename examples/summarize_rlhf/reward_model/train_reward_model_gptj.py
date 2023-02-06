import os

import torch
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

def remap_chosen_rejected_to_ranking(dataset):
    result = []
    for sample in tqdm(dataset):
        result.append({
            "prompt": sample["prompt"],
            "ranked_outputs": [
                sample["chosen"],
                sample["rejected"]
            ]
        })
    return result


def remove_duplicates(xs):
    # keys are inserted in order, so only the first occurrence of each element in each list is preserved
    return list(dict.fromkeys(xs))


def is_too_short(output):
    return len(output.split()) < 5


def create_comparison_dataset(dataset):
    result = []
    for sample in tqdm(dataset):
        ranked_outputs = remove_duplicates([
            f"{sample['prompt']}\n{output}"
            for output in sample["ranked_outputs"]
            if not is_too_short(output)]
        )
        if len(ranked_outputs) >= 2:
            result.append(ranked_outputs)
    return result


class PairwiseDataset(Dataset):
    def __init__(self, rankings, tokenizer, max_length):
        self.items = []
        for ranking in tqdm(rankings):
            current_items = []
            for output in ranking:
                encodings_dict = tokenizer(
                    "<|startoftext|>" + output + "<|endoftext|>",
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                current_items.append(encodings_dict["input_ids"])
                current_items.append(encodings_dict["attention_mask"])
            self.items.append(current_items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat(sum([[x[k] for k in range(len(x)) if k % 2 == 0] for x in data], []))
        batch["attention_mask"] = torch.cat(sum([[x[k] for k in range(len(x)) if k % 2 == 1] for x in data], []))
        batch["labels"] = torch.tensor(sum([[i] * len(data) for i in range(len(data))], []))

        batch2 = {}
        batch2["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch2["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch2["labels"] = torch.tensor([0] * len(data) + [1] * len(data))

        assert batch == batch2

        return batch


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists("rm_checkpoint"):
        os.mkdir("rm_checkpoint")

    training_args = TrainingArguments(
        output_dir="rm_checkpoint/",
        num_train_epochs=5,
        logging_steps=10,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        deepspeed="ds_config_gpt_j.json",
        save_total_limit=1,
    )

    # Create the comparisons datasets
    data_path = "CarperAI/openai_summarize_comparisons"
    dataset = load_dataset(data_path)
    train_pairs = create_comparison_dataset(remap_chosen_rejected_to_ranking(dataset["train"]))
    val_pairs = create_comparison_dataset(remap_chosen_rejected_to_ranking(dataset["test"]))

    # Make pairwise datasets for training
    max_length = 550
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model = GPTRewardModel("CarperAI/openai_summarize_tldr_sft")

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    ).train()
