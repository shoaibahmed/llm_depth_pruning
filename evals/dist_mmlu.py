import os
import sys
import json
import wandb
import pickle
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer

from .mmlu import gen_prompt, format_example
from .mmlu_utils import download_mmlu

sys.path.append("..")  # top-level package
from dist_utils import is_main_proc, wait_for_other_procs, reduce_tensor


class MMLUDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, model_name: str, dataset_path: str = None, n_train: int = 5):
        if dataset_path is None:
            dataset_path = "./datasets/mmlu/"  # default dataset directory
            if not os.path.exists(dataset_path) and is_main_proc():
                MMLUDataset.download_dataset()
            wait_for_other_procs()
        assert os.path.exists(dataset_path)
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.n_train = n_train
        self.tokenized_dataset_path = os.path.join(dataset_path, f"{model_name}_tokenized_ntrain_{n_train}.pkl")

        # Create the dataset containers
        self.prompts = []
        self.labels = []
        self.subjects = []
        self.load_mmlu(tokenizer)
        self.label2idx = {char: i for i, char in enumerate(["A", "B", "C", "D"])}

    def load_mmlu(self, tokenizer: PreTrainedTokenizer):
        if not self.is_dataset_processed():
            if is_main_proc():
                self.tokenize_mmlu(tokenizer, self.n_train)
                self.save_dataset()
            wait_for_other_procs()
        self.load_dataset()

    def tokenize_mmlu(self, tokenizer: PreTrainedTokenizer, n_train: int):
        subjects = sorted(
            [
                f.split("_test.csv")[0]
                for f in os.listdir(os.path.join(self.dataset_path, "test"))
                if "_test.csv" in f
            ]
        )

        for subject in subjects:
            dev_df = pd.read_csv(
                os.path.join(self.dataset_path, "dev", subject + "_dev.csv"), header=None
            )[:n_train]
            test_df = pd.read_csv(
                os.path.join(self.dataset_path, "test", subject + "_test.csv"), header=None
            )

            for i in range(test_df.shape[0]):
                # get prompt and make sure it fits
                k = n_train
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end

                input_ids = tokenizer(prompt, return_tensors="pt").input_ids

                while input_ids.shape[-1] > 2048:
                    k -= 1
                    train_prompt = gen_prompt(dev_df, subject, k)
                    prompt = train_prompt + prompt_end
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                label = test_df.iloc[i, test_df.shape[1] - 1]

                # Add to the list
                assert input_ids.shape[0] == 1, input_ids.shape
                self.prompts.append(input_ids[0, :].clone())
                self.labels.append(label)
                self.subjects.append(subject)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"input_ids": self.prompts[idx], "label": self.label2idx[self.labels[idx]],
                "subject": self.subjects[idx]}

    @staticmethod
    def download_dataset(dataset_dir=None):
        download_mmlu(dataset_dir)

    def is_dataset_processed(self):
        return os.path.exists(self.tokenized_dataset_path)

    def save_dataset(self):
        if not self.is_dataset_processed():
            print("Saving MMLU dataset to disk:", self.tokenized_dataset_path)
            with open(self.tokenized_dataset_path, "wb") as f:
                output_dict = {"prompts": self.prompts, "labels": self.labels, "subjects": self.subjects}
                pickle.dump(output_dict, f)

    def load_dataset(self):
        assert self.is_dataset_processed()
        print("Loading MMLU dataset from disk:", self.tokenized_dataset_path)
        with open(self.tokenized_dataset_path, "rb") as f:
            output_dict = pickle.load(f)
            self.prompts = output_dict["prompts"]
            self.labels = output_dict["labels"]
            self.subjects = output_dict["subjects"]


@torch.no_grad()
def evaluate_mmlu(model: torch.nn.Module, tokenizer: PreTrainedTokenizer, mmlu_loader: torch.utils.data.DataLoader,
                  device: torch.device, split_name: str, verbose: bool = False):
    total = 0
    correct = 0
    tokenized_ids = {char: tokenizer(char).input_ids[-1] for char in ["A", "B", "C", "D"]}
    if verbose:
        print("Tokenized IDs:", tokenized_ids)

    for batch in tqdm(mmlu_loader):
        input_ids = batch["input_ids"].to(device)
        assert input_ids.shape[0] == 1, input_ids.shape

        # Forward prop through the model
        logits = model(input_ids=input_ids).logits
        assert logits.shape[0] == 1, f"batch size should be 1. Found: {logits.shape}"
        logits = logits[:, -1, :].flatten()  # BSV format

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenized_ids["A"]],
                        logits[tokenized_ids["B"]],
                        logits[tokenized_ids["C"]],
                        logits[tokenized_ids["D"]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = np.argmax(probs)  # idx of the correct option
        is_correct = int(pred) == int(batch["label"])
        correct += int(is_correct)
        if verbose:
            print(f"probs: {probs} / pred: {pred} / label: {int(batch['label'])} / correct: {is_correct}")
        total += 1

    correct = int(reduce_tensor(torch.tensor(correct).to(device)))
    total = int(reduce_tensor(torch.tensor(total).to(device)))
    weighted_acc = 100. * correct / total

    output_dict = {"mmlu_split": split_name, "total": total, "correct": correct, "weighted_acc": weighted_acc}
    print(json.dumps(output_dict))
    if split_name is not None and wandb.run is not None:
        wandb.log({f"mmlu_{split_name}": {"total": total, "correct": correct, "weighted_acc": weighted_acc}})
    return correct, total, weighted_acc


if __name__ == "__main__":
    from argparse import Namespace
    from trainer import load_model
    from dataset import get_dataloader
    from dist_utils import init_distributed_env

    model_name = 'llama-2'
    args = Namespace(model_name=model_name, model_size='7b', use_instruct_model=False,
                     amp_dtype=None, use_gradient_checkpointing=False, batch_size=1,
                     num_workers=8, seed=43)

    # Setup the distributed env
    init_distributed_env(args)

    # Load the tokenizer
    tokenizer = load_model(args, only_tokenizer=True)

    # Wrap the dataset into a dataloader
    dataset = MMLUDataset(tokenizer, model_name)
    print("# examples in dataset:", len(dataset))

    # Generator to seed dataloaders
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dl = get_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        generator=generator)  # no padding so batch size should be 1

    for i, input_dict in enumerate(dl):
        print("Prompt:", input_dict["input_ids"].shape, input_dict["input_ids"][:10])
        print("Label:", input_dict["label"])
        print("Subject:", input_dict["subject"])
        print("="*10)
        if i >= 5:
            break

    # Test evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model(args, pretrained=True)
    model.eval()  # set the model in eval mode
    model = model.to(device) # move the model to device
    correct, total, weighted_acc = evaluate_mmlu(model, tokenizer, dl, device, split_name=None, verbose=True)
    print(f"Final weighted acc: {weighted_acc:.2f}% ({correct}/{total})")
