#!/bin/python

import os
import time
import json
import random
import argparse
from tqdm import tqdm
from typing import Tuple

import wandb

import torch
import numpy as np

from transformers import AutoTokenizer, AutoConfig

import sys
sys.path.append('.')
from llama_model import LlamaForCausalLM
from mistral_model import MistralForCausalLM
from dataset import NLPDataset, get_dataloader
from train_utils import get_num_model_params
from dist_utils import init_distributed_env, is_main_proc, wait_for_other_procs, reduce_tensor
from block_influence import BlockInfluenceEstimator
from evals.dist_mmlu import MMLUDataset, evaluate_mmlu


def load_model(args, only_tokenizer=False, pretrained=False):
    # assumes huggingface login: `huggingface-cli login``
    if args.model_name == "llama-2":
        if args.use_instruct_model:
            model_name = f"meta-llama/Llama-2-{args.model_size.lower()}-chat-hf"
        else:
            model_name = f"meta-llama/Llama-2-{args.model_size.lower()}-hf"
    elif args.model_name == "mistral":
        if args.use_instruct_model:
            model_name = f"mistralai/Mistral-{args.model_size.upper()}-Instruct-v0.2"
        else:
            model_name = f"mistralai/Mistral-{args.model_size.upper()}-v0.1"
    else:
        raise RuntimeError(f"Unsupported model: {args.model_name}")
    print("!! Loading model:", model_name)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if only_tokenizer:
        return tokenizer

    # Load the model as well as the tokenizer
    config = AutoConfig.from_pretrained(model_name)
    print("Config:", config)
    kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    print("Model precision:", kwargs["torch_dtype"])
    if pretrained:
        print("Using pretrained model...")

    if args.model_name == "llama-2":
        if not pretrained:
            model = LlamaForCausalLM(config).to(kwargs["torch_dtype"])
        else:
            model = LlamaForCausalLM.from_pretrained(model_name, **kwargs)
    elif args.model_name == "mistral":
        if not pretrained:
            model = MistralForCausalLM(config).to(kwargs["torch_dtype"])
        else:
            model = MistralForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        raise RuntimeError(f"Unsupported model: {args.model_name}")
    return model, tokenizer


def compute_log_probs(logits: torch.Tensor, target_ids: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    # Apply softmax and log to obtain log probabilities from logits (summing original logits would be incorrect)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    sequence_log_prob = log_probs.sum(dim=1).cpu().float().numpy()

    # Calculate perplexity
    sequence_length = target_ids.size(-1)
    assert sequence_length > 0, logits
    sequence_perplexity = np.exp(-sequence_log_prob / sequence_length)

    return sequence_perplexity, sequence_log_prob


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, eval_loader: torch.utils.data.DataLoader, device: torch.device, split_name: str):
    model.eval()
    avg_sequence_perplexity = 0.
    avg_loss = 0.
    num_ex = 0

    for batch in tqdm(eval_loader):
        tokenized_input = batch["input_ids"].to(device)

        # Forward prop through the model (will also populate the loss, but one extra logit)
        outputs = model(tokenized_input, labels=tokenized_input)

        # Compute metrics on top of LM logits
        lm_logits = outputs.logits[:, :-1, :]  # BTD format (discard the final logit)
        target_ids = tokenized_input[:, 1:]  # input ids strided by one
        assert len(lm_logits.shape) == 3, lm_logits.shape
        assert len(target_ids.shape) == 2, target_ids.shape
        assert lm_logits.shape[1] == target_ids.shape[1], f"{lm_logits.shape} != {target_ids.shape}"
        perplexity, log_prob = compute_log_probs(lm_logits, target_ids)

        avg_sequence_perplexity += float(perplexity.sum())
        avg_loss += float(outputs.loss)
        num_ex += len(tokenized_input)

    # Collect the stats from all processes
    avg_sequence_perplexity = float(reduce_tensor(torch.tensor(avg_sequence_perplexity).to(device)))
    avg_loss = float(reduce_tensor(torch.tensor(avg_loss).to(device)))
    num_ex = int(reduce_tensor(torch.tensor(num_ex).to(device)))

    avg_sequence_perplexity = avg_sequence_perplexity / num_ex
    avg_loss = avg_loss / num_ex
    output_dict = {"split": split_name, "num_ex": num_ex, "avg_loss": avg_loss, "avg_seq_perplexity": avg_sequence_perplexity}
    print(json.dumps(output_dict))
    if split_name is not None and wandb.run is not None:
        wandb.log({f"eval_{split_name}": {"num_ex": num_ex, "avg_loss": avg_loss, "avg_seq_perplexity": avg_sequence_perplexity}})
    return avg_loss, avg_sequence_perplexity


@torch.no_grad()
def compute_block_shapley(model: torch.nn.Module, eval_loader: torch.utils.data.DataLoader, device: torch.device,
                          use_random_subnetworks: bool = False, subnetwork_len: float = 0.5, seed: int = 43,
                          num_subsampled_networks: int = 10, max_samples_per_proc: int = None):
    model.eval()
    num_model_layers = model.get_num_model_layers()
    print(f"!! Computing the logit shapley value for the model with {num_model_layers} layers...")
    rng = np.random.default_rng(seed)
    if not use_random_subnetworks:
        num_subsampled_networks = num_model_layers

    all_statistics = []
    for iterator, batch in enumerate(tqdm(eval_loader)):
        tokenized_input = batch["input_ids"].to(device)
        base_logits = None
        for i in range(1+num_subsampled_networks):  # first one is always base model eval
            selected_blocks = None  # use full network
            if i != 0:  # use subnetwork
                if use_random_subnetworks:
                    selected_blocks = rng.choice(range(num_model_layers), int(subnetwork_len*num_model_layers), replace=False)
                else:
                    block_to_remove = i - 1
                    selected_blocks = [x for x in range(num_model_layers) if x != block_to_remove]
            model.select_blocks(selected_blocks, verbose=False)

            outputs = model(tokenized_input, labels=tokenized_input)
            lm_logits = outputs.logits[:, :-1, :]  # BTD format (discard the final logit)
            lm_loss = outputs.loss
            if base_logits is None:
                assert selected_blocks is None
                base_logits = lm_logits
            else:
                assert selected_blocks is not None
                diff_norm = torch.norm(base_logits - lm_logits, p=2, dim=-1).mean()  # mean over batch and sequence
                all_statistics.append((selected_blocks, float(diff_norm), float(lm_loss)))

        # Check if stopping condition is met
        if max_samples_per_proc is not None and iterator >= max_samples_per_proc - 1:
            print(f"{iterator} samples collected for logit shapley value. Stopping further computations!")
            break

    # Compute the block influence based on the computed statistics
    logit_dist = {i: {"present": [], "absent": []} for i in range(num_model_layers)}
    loss_dist = {i: {"present": [], "absent": []} for i in range(num_model_layers)}
    for selected_blocks, diff_norm, loss in all_statistics:
        for i in range(num_model_layers):
            key = "present" if i in selected_blocks else "absent"
            logit_dist[i][key].append(diff_norm)
            loss_dist[i][key].append(loss)

    # Compute average distances
    print("~~~~~~ Block shapley statistics ~~~~~~")
    logit_shapley_list = []
    loss_shapley_list = []
    for key, input_container, output_container in [("dist", logit_dist, logit_shapley_list),
                                                   ("loss", loss_dist, loss_shapley_list)]:
        for i in range(num_model_layers):
            for name in ["present", "absent"]:
                mean = np.mean(input_container[i][name])  # convert it to mean
                input_container[i][name] = float(reduce_tensor(torch.tensor(mean).to(device), average=True))
            shapley = input_container[i]['present'] - input_container[i]['absent']
            print(f"> block {i} / present mean {key}: {input_container[i]['present']} / absent mean {key}: {input_container[i]['absent']} / shapley: {shapley}")
            output_container.append(shapley)
    print("-"*50)
    return logit_shapley_list, loss_shapley_list


def main(args):
    init_distributed_env(args)

    generator = None
    if args.seed is not None:  # Set process seed to reduce stochasticity
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(seed=args.seed)
        random.seed(args.seed)
        print("Setting process seed:", args.seed)

        # Generator to seed dataloaders
        generator = torch.Generator()
        generator.manual_seed(args.seed)

    dataset_dir = f"{args.dataset}_model_{args.model_name}_seq_len_{args.sequence_length}_subsample_{args.subsample_size}_comb_docs"
    args.dataset_output_dir = os.path.join("datasets", dataset_dir)

    suffix = "block_pruning"
    args.wandb_run_name = f"{dataset_dir}_{suffix}"

    if args.wandb_project is not None and is_main_proc():
        print("Initialization w&b...")
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args, resume=False)

    if is_main_proc() and not NLPDataset.is_dataset_processed(args.dataset_output_dir):
        tokenizer = load_model(args, only_tokenizer=True)
        dataset = NLPDataset(args.dataset, tokenizer, max_length=args.sequence_length,
                             combine_documents=True, subsample_size=args.subsample_size)
        dataset.save_datasets(args.dataset_output_dir)
    wait_for_other_procs()  # wait for the main process to write the dataset

    # Load the dataset
    dataset = NLPDataset.load_dataset(args.dataset_output_dir)  # returns a dataset dict
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Load the model
    model, tokenizer = load_model(args, pretrained=True)
    num_model_params = get_num_model_params(model)
    num_model_layers = model.get_num_model_layers()
    print(f"# model params: {num_model_params/1_000_000:.2f}M / # layers: {num_model_layers}")

    # Convert to DDP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # move to device

    # Create the dataloaders
    train_loader = get_dataloader(train_dataset, args.batch_size, args.num_workers, drop_last=True, generator=generator)
    eval_loader = get_dataloader(test_dataset, args.test_batch_size, args.num_workers, generator=generator)

    # Load MMLU dataset
    mmlu_dataset = MMLUDataset(tokenizer, args.model_name)
    print("# examples in MMLU dataset:", len(mmlu_dataset))
    mmlu_loader = get_dataloader(mmlu_dataset, 1, args.num_workers, generator=generator)  # bs=1 for MMLU

    print(">> Estimating block influences...")
    eval_start_time = time.time()
    model.select_blocks(None)  # use all blocks
    block_influence_estimator = BlockInfluenceEstimator(num_model_layers, device)
    model.add_block_influence_estimator(block_influence_estimator)
    evaluate_model(model, train_loader, device, split_name=None)  # use the train set to compute block influences
    final_block_influences = block_influence_estimator.get_block_influences()
    model.add_block_influence_estimator(None)  # remove the block influence computation
    print("Final block influences:", final_block_influences)

    cosine_block_influences = [x["cosine_dist"] for x in final_block_influences]
    l1_block_influences = [x["l1_update_norm"] for x in final_block_influences]
    relative_l1_block_influences = [x["l1_relative_update_norm"] for x in final_block_influences]
    l2_block_influences = [x["l2_update_norm"] for x in final_block_influences]
    relative_l2_block_influences = [x["l2_relative_update_norm"] for x in final_block_influences]
    print("Cosine block influences:", cosine_block_influences)
    print("L1 block influences:", l1_block_influences)
    print("Relative L1 block influences:", relative_l1_block_influences)
    print("L2 block influences:", l2_block_influences)
    print("Relative L2 block influences:", relative_l2_block_influences)

    if wandb.run is not None:
        wandb.log({f"block_{i}_cosine_influence": block_influence for i, block_influence in enumerate(cosine_block_influences)})
        wandb.log({f"block_{i}_l1_influence": block_influence for i, block_influence in enumerate(l1_block_influences)})
        wandb.log({f"block_{i}_relative_l1_influence": block_influence for i, block_influence in enumerate(relative_l1_block_influences)})
        wandb.log({f"block_{i}_l2_influence": block_influence for i, block_influence in enumerate(l2_block_influences)})
        wandb.log({f"block_{i}_relative_l2_influence": block_influence for i, block_influence in enumerate(relative_l2_block_influences)})

    # Compute the block logit shapley
    max_samples_per_proc = None
    if args.limit_shapley_samples is not None:
        max_samples_per_proc = args.limit_shapley_samples // args.world_size
        print(f"Total samples limit: {args.limit_shapley_samples}. Capping the max_samples_per_proc for logit shapley computation to be: {max_samples_per_proc}")
    block_logit_shapley, block_loss_shapley = compute_block_shapley(model, train_loader, device, max_samples_per_proc=max_samples_per_proc)
    print("Block logit shapley:", block_logit_shapley)
    print("Block loss shapley:", block_loss_shapley)
    logit_shapley_block_influence = [-x for x in block_logit_shapley]  # negative shapely (lower distance) indicates higher importance
    loss_shapley_block_influence = [-x for x in block_loss_shapley]  # negative shapely (lower distance) indicates higher importance
    if wandb.run is not None:
        wandb.log({f"block_{i}_logit_shapley_influence": block_influence for i, block_influence in enumerate(logit_shapley_block_influence)})
        wandb.log({f"block_{i}_loss_shapley_influence": block_influence for i, block_influence in enumerate(loss_shapley_block_influence)})

    block_influence_list = [("cosine", cosine_block_influences), ("relative_l1", relative_l1_block_influences),
                            ("relative_l2", relative_l2_block_influences), ("logit_shapley", logit_shapley_block_influence),
                            ("loss_shapley", loss_shapley_block_influence)]
    for influence_name, block_influences in block_influence_list:
        print("Using block influence method:", influence_name)
        print("Block influence values:", block_influences)
        sorted_blocks = np.argsort(block_influences)  # ascending order
        print("Sorted block list:", sorted_blocks)

        remaining_blocks = list(range(num_model_layers))
        weighted_acc_list = []
        perplexity_list = []
        iterator = -1
        for _ in range(len(sorted_blocks)+1):  # one additional iteration for no dropping
            if iterator > -1:  # do nothing for the first block i.e., all blocks are selected
                lowest_block = sorted_blocks[iterator]  # prune blocks based on the estimated block influence
                print(f"Removing block {lowest_block} with lowest influence: {block_influences[lowest_block]}")
                remaining_blocks = [i for i in remaining_blocks if i != lowest_block]  # remove lowest block
            print("Remaining blocks:", remaining_blocks)
            model.select_blocks(remaining_blocks)  # use all blocks
            _, _, weighted_acc = evaluate_mmlu(model, tokenizer, mmlu_loader, device, f"{influence_name}_blocks_pruned_{iterator+1}")
            _, avg_perplexity = evaluate_model(model, eval_loader, device, f"{influence_name}_blocks_pruned_{iterator+1}")
            weighted_acc_list.append(weighted_acc)
            perplexity_list.append(avg_perplexity)
            iterator += 1

        print(f">>>>> Block pruning statistics using {influence_name} metric <<<<<")
        print(f"{influence_name} weighted ACC list: {weighted_acc_list}")
        print(f"{influence_name} perplexity list: {perplexity_list}")
        print("="*25)

    eval_time_elapsed_h = (time.time() - eval_start_time) / (60 * 60)  # convert seconds into hours
    print(f"Block pruning evaluation completed / time elapsed: {eval_time_elapsed_h:.2f}h")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    supported_datasets = ['pg19', 'cc_news', 'wikitext-2', 'bookcorpus', 'c4', 'openwebtext', 'slimpajama']

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Argument parser for LLM block influence evaluator')

    # Add arguments
    parser.add_argument('-d', '--dataset', default='wikitext-2', choices=supported_datasets,
                        help='Dataset name (default: wikitext-2)')
    parser.add_argument('-m', '--model-name', default='llama-2', choices=['llama-2', 'mistral'],
                        help='Model name (default: llama-2)')
    parser.add_argument('-s', '--model-size', default='7b', choices=['7b'],
                        help='Model size (default: 7b)')
    parser.add_argument('--use-instruct-model', action='store_true', default=False,
                        help='Use instruction-tuned model rather than the base model')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size per process (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        help='Batch size per process for testing (default: equal to --batch-size)')
    parser.add_argument('--sequence-length', type=int, default=1024,
                        help='Sequence length for computing the model perplexity (default: 1024)')
    parser.add_argument('--subsample-size', type=int, default=1000000,
                        help='Dataset subsample size in terms of number of docs (default: 1M)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for the dataloader (default: 8)')
    parser.add_argument('--seed', type=int, default=43,
                        help='seed value (default: 43)')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='W&B project name (none indicates no W&B initialization)')
    parser.add_argument('--limit-shapley-samples', type=int, default=None,
                        help='limit the number of samples to the specified value for shapley computation (default: None i.e., no limit)')

    # Parse the arguments
    args = parser.parse_args()

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
        print("Setting test batch size to be equal to batch size:", args.test_batch_size)

    main(args)
