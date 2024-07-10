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

from llama_layer_influence import LlamaForCausalLM
from mistral_layer_influence import MistralForCausalLM

import sys
sys.path.append('.')
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
def compute_layer_shapley(model: torch.nn.Module, eval_loader: torch.utils.data.DataLoader, device: torch.device,
                          use_random_subnetworks: bool = False, subnetwork_len: float = 0.5, seed: int = 43,
                          num_subsampled_networks: int = 10, max_samples_per_proc: int = None):
    model.eval()
    base_module = model.module if hasattr(model, 'module') else model  # redefine the base module
    num_model_layers = base_module.get_num_model_layers()
    num_model_layers = num_model_layers * 2  # MHSA + MLP in each block
    print(f"!! Computing the logit shapley value for the model with {num_model_layers} layers...")
    rng = np.random.default_rng(seed)
    if not use_random_subnetworks:
        num_subsampled_networks = num_model_layers

    all_statistics = []
    for iterator, batch in enumerate(tqdm(eval_loader)):
        tokenized_input = batch["input_ids"].to(device)
        base_logits = None
        for i in range(1+num_subsampled_networks):  # first one is always base model eval
            selected_layers = None  # use full network
            if i != 0:  # use subnetwork
                if use_random_subnetworks:
                    selected_layers = rng.choice(range(num_model_layers), int(subnetwork_len*num_model_layers), replace=False)
                else:
                    layer_to_remove = i - 1
                    selected_layers = [x for x in range(num_model_layers) if x != layer_to_remove]
            base_module.select_layers(selected_layers, verbose=False)

            outputs = model(tokenized_input, labels=tokenized_input)
            lm_logits = outputs.logits[:, :-1, :]  # BTD format (discard the final logit)
            lm_loss = outputs.loss
            if base_logits is None:
                assert selected_layers is None
                base_logits = lm_logits
            else:
                assert selected_layers is not None
                diff_norm = torch.norm(base_logits - lm_logits, p=2, dim=-1).mean()  # mean over batch and sequence
                all_statistics.append((selected_layers, float(diff_norm), float(lm_loss)))

        # Check if stopping condition is met
        if max_samples_per_proc is not None and iterator >= max_samples_per_proc - 1:
            print(f"{iterator} samples collected for logit shapley value. Stopping further computations!")
            break

    # Compute the layer influence based on the computed statistics
    logit_dist = {i: {"present": [], "absent": []} for i in range(num_model_layers)}
    loss_dist = {i: {"present": [], "absent": []} for i in range(num_model_layers)}
    for selected_layers, diff_norm, loss in all_statistics:
        for i in range(num_model_layers):
            key = "present" if i in selected_layers else "absent"
            logit_dist[i][key].append(diff_norm)
            loss_dist[i][key].append(loss)

    # Compute average distances
    print("~~~~~~ Layer shapley statistics ~~~~~~")
    logit_shapley_list = []
    loss_shapley_list = []
    for key, input_container, output_container in [("dist", logit_dist, logit_shapley_list),
                                                   ("loss", loss_dist, loss_shapley_list)]:
        for i in range(num_model_layers):
            for name in ["present", "absent"]:
                mean = np.mean(input_container[i][name])  # convert it to mean
                input_container[i][name] = float(reduce_tensor(torch.tensor(mean).to(device), average=True))
            shapley = input_container[i]['present'] - input_container[i]['absent']
            print(f"> layer {i} / present mean {key}: {input_container[i]['present']} / absent mean {key}: {input_container[i]['absent']} / shapley: {shapley}")
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

    suffix = f"layer_pruning_{args.pruning_scheme}"
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

    # Create the low-rank adapters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # move to device

    # Create the dataloaders
    train_loader = get_dataloader(train_dataset, args.batch_size, args.num_workers, drop_last=True, generator=generator)
    eval_loader = get_dataloader(test_dataset, args.test_batch_size, args.num_workers, generator=generator)

    # Load MMLU dataset
    mmlu_dataset = MMLUDataset(tokenizer, args.model_name)
    print("# examples in MMLU dataset:", len(mmlu_dataset))
    mmlu_loader = get_dataloader(mmlu_dataset, 1, args.num_workers, generator=generator)  # bs=1 for MMLU

    eval_start_time = time.time()
    model.select_layers(None)  # use all layers
    total_layers = num_model_layers * 2  # MHSA + MLP in each block
    print("Total model layers:", total_layers)
    if args.use_block_influence:
        print(">> Using block influences for layer pruning...")
        if args.model_name == "llama-2":
            cosine_block_influences = [0.4065510630607605, 0.24990439414978027, 0.1453840732574463, 0.1462407112121582, 0.1515178084373474, 0.15010249614715576, 0.1436394453048706, 0.14198929071426392, 0.1338726282119751, 0.1287376880645752, 0.12480568885803223, 0.11314243078231812, 0.10922092199325562, 0.10999512672424316, 0.10579818487167358, 0.10590404272079468, 0.10621917247772217, 0.08393210172653198, 0.07559847831726074, 0.06368941068649292, 0.06160825490951538, 0.046613335609436035, 0.04402130842208862, 0.036509573459625244, 0.0347287654876709, 0.030786514282226562, 0.033209800720214844, 0.02987152338027954, 0.03320187330245972, 0.033727407455444336, 0.06028544902801514, 0.3071935772895813]
        else:
            assert args.model_name == "mistral"
            cosine_block_influences = [0.4131488800048828, 0.22294247150421143, 0.18501925468444824, 0.15420764684677124, 0.15791428089141846, 0.16193944215774536, 0.15579771995544434, 0.15514129400253296, 0.14191526174545288, 0.13151496648788452, 0.12524676322937012, 0.1169435977935791, 0.11011826992034912, 0.10614091157913208, 0.11162376403808594, 0.11073744297027588, 0.1094699501991272, 0.10527843236923218, 0.10619938373565674, 0.09696263074874878, 0.07502329349517822, 0.06127279996871948, 0.04937338829040527, 0.04653525352478027, 0.04156315326690674, 0.03952103853225708, 0.03852170705795288, 0.04080432653427124, 0.050363004207611084, 0.06016743183135986, 0.06128782033920288, 0.17413002252578735]
        cosine_layer_influences = []
        for inf in cosine_block_influences:
            cosine_layer_influences += [inf, inf]  # MHSA / MLP
        assert len(cosine_layer_influences) == total_layers, f"{len(cosine_layer_influences)} != {total_layers}"
        layer_influence_list = [("block_cosine", cosine_layer_influences)]
        assert args.pruning_scheme in ["mhsa", "mlp"]
    else:
        print(">> Estimating layer influences...")
        block_influence_estimator = BlockInfluenceEstimator(total_layers, device)
        model.add_layer_influence_estimator(block_influence_estimator)
        evaluate_model(model, train_loader, device, split_name=None)  # use the train set to compute block influences
        final_layer_influences = block_influence_estimator.get_block_influences()
        model.add_layer_influence_estimator(None)  # remove the block influence computation
        print("Final layer influences:", final_layer_influences)

        cosine_layer_influences = [x["cosine_dist"] for x in final_layer_influences]
        l1_layer_influences = [x["l1_update_norm"] for x in final_layer_influences]
        relative_l1_layer_influences = [x["l1_relative_update_norm"] for x in final_layer_influences]
        l2_layer_influences = [x["l2_update_norm"] for x in final_layer_influences]
        relative_l2_layer_influences = [x["l2_relative_update_norm"] for x in final_layer_influences]
        print("Cosine layer influences:", cosine_layer_influences)
        print("L1 layer influences:", l1_layer_influences)
        print("Relative L1 layer influences:", relative_l1_layer_influences)
        print("L2 layer influences:", l2_layer_influences)
        print("Relative L2 layer influences:", relative_l2_layer_influences)

        if wandb.run is not None:
            wandb.log({f"layer_{i}_cosine_influence": layer_influence for i, layer_influence in enumerate(cosine_layer_influences)})
            wandb.log({f"layer_{i}_l1_influence": layer_influence for i, layer_influence in enumerate(l1_layer_influences)})
            wandb.log({f"layer_{i}_relative_l1_influence": layer_influence for i, layer_influence in enumerate(relative_l1_layer_influences)})
            wandb.log({f"layer_{i}_l2_influence": layer_influence for i, layer_influence in enumerate(l2_layer_influences)})
            wandb.log({f"layer_{i}_relative_l2_influence": layer_influence for i, layer_influence in enumerate(relative_l2_layer_influences)})

        # Compute the block logit shapley
        max_samples_per_proc = None
        if args.limit_shapley_samples is not None:
            max_samples_per_proc = args.limit_shapley_samples // args.world_size
            print(f"Total samples limit: {args.limit_shapley_samples}. Capping the max_samples_per_proc for logit shapley computation to be: {max_samples_per_proc}")
        layer_logit_shapley, layer_loss_shapley = compute_layer_shapley(model, train_loader, device, max_samples_per_proc=max_samples_per_proc)
        print("Layer logit shapley:", layer_logit_shapley)
        print("Layer loss shapley:", layer_loss_shapley)
        logit_shapley_layer_influence = [-x for x in layer_logit_shapley]  # negative shapely (lower distance) indicates higher importance
        loss_shapley_layer_influence = [-x for x in layer_loss_shapley]    # negative shapely (lower distance) indicates higher importance
        if wandb.run is not None:
            wandb.log({f"layer_{i}_logit_shapley_influence": layer_influence for i, layer_influence in enumerate(logit_shapley_layer_influence)})
            wandb.log({f"layer_{i}_loss_shapley_influence": layer_influence for i, layer_influence in enumerate(loss_shapley_layer_influence)})

        layer_influence_list = [("cosine", cosine_layer_influences), ("relative_l1", relative_l1_layer_influences),
                                ("relative_l2", relative_l2_layer_influences), ("logit_shapley", logit_shapley_layer_influence),
                                ("loss_shapley", loss_shapley_layer_influence)]

    for influence_name, layer_influences in layer_influence_list:
        print("Using layer influence method:", influence_name)
        print("Layer influence values:", layer_influences)
        sorted_layers = np.argsort(layer_influences)  # ascending order
        if args.pruning_scheme != "both":
            print("Selected pruning scheme:", args.pruning_scheme)
            if args.pruning_scheme == "mhsa":
                print("Only keeping even layers for removal...")
                sorted_layers = [x for x in sorted_layers if x % 2 == 0]  # MHSA layers are even
            else:
                assert args.pruning_scheme == "mlp", args.pruning_scheme
                print("Only keeping odd layers for removal...")
                sorted_layers = [x for x in sorted_layers if x % 2 == 1]  # MLP layers are odd
        print("Sorted layer list:", sorted_layers)

        remaining_layers = list(range(total_layers))
        weighted_acc_list = []
        perplexity_list = []
        iterator = -1
        for _ in range(len(sorted_layers)+1):  # one additional iteration for no dropping
            if iterator > -1:  # do nothing for the first layer i.e., all layers are selected
                lowest_layer = sorted_layers[iterator]  # prune blocks based on the estimated block influence
                print(f"Removing layer {lowest_layer} with lowest influence: {layer_influences[lowest_layer]}")
                remaining_layers = [i for i in remaining_layers if i != lowest_layer]  # remove lowest layer
            print("Remaining layers:", remaining_layers)
            model.select_layers(remaining_layers)  # use the selected layers
            _, _, weighted_acc = evaluate_mmlu(model, tokenizer, mmlu_loader, device, f"{influence_name}_layers_pruned_{iterator+1}")
            _, avg_perplexity = evaluate_model(model, eval_loader, device, f"{influence_name}_layers_pruned_{iterator+1}")
            weighted_acc_list.append(weighted_acc)
            perplexity_list.append(avg_perplexity)
            iterator += 1

        print(f">>>>> Layer pruning statistics using {influence_name} metric <<<<<")
        print(f"{influence_name} weighted ACC list: {weighted_acc_list}")
        print(f"{influence_name} perplexity list: {perplexity_list}")
        print("="*25)

    eval_time_elapsed_h = (time.time() - eval_start_time) / (60 * 60)  # convert seconds into hours
    print(f"Layer pruning evaluation completed / time elapsed: {eval_time_elapsed_h:.2f}h")

    if wandb.run is not None:
        wandb.finish()
    print("Script execution completed!")


if __name__ == "__main__":
    supported_datasets = ['pg19', 'cc_news', 'wikitext-2', 'bookcorpus', 'c4', 'openwebtext', 'slimpajama']

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Argument parser for LLM layer influence evaluator')

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
    parser.add_argument('--pruning-scheme', default='both', choices=['mhsa', 'mlp', 'both'],
                        help='Pruning scheme (default: both)')
    parser.add_argument('--limit-shapley-samples', type=int, default=None,
                        help='limit the number of samples to the specified value for shapley computation (default: None i.e., no limit)')
    parser.add_argument('--use-block-influence', action='store_true', default=False,
                        help='use block influence for deciding the layer importance rather than computing it')

    # Parse the arguments
    args = parser.parse_args()

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
        print("Setting test batch size to be equal to batch size:", args.test_batch_size)

    main(args)
