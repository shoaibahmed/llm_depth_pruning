# A deeper look at depth pruning of LLMs

The official implementation for the paper: "A deeper look at depth pruning of LLMs" (https://arxiv.org/abs/).


## Usage

The main scripts are divided into three different categories.

### Block influence

The main experiments are based on block influence where we compute the impact of different block influence estimation techniques for block pruning.
The associated scripts are located in the directory: `./block_influence/`.
The evaluate the model based on different block influence techniques, check the script: `block_influence/eval.sh`.

### MMLU Shapley

MMLU shapley focuses on computing the loss shapley directly on the MMLU test set, which serves as an upperbound on the performance that can be achieved by different block pruning techniques on the MMLU benchmark.
The experiments associated with MMLU shapley based block pruning are located in the directory: `./block_influence_mmlu_shapley/`.
The evaluate the model based on MMLU loss shapley based block pruning, check the script: `block_influence_mmlu_shapley/eval.sh`.

### Layer influence

We further disect the transformer block into its corresponding feed-forward and self-attention layers, and evaluate their impact separately.
The experiments associated with layer influence estimation are located in the directory: `./layer_influence/`.
The evaluate the model based on different layer influence techniques, check the script: `layer_influence/eval.sh`.


## Citation


## License

MIT
