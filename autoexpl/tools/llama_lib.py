# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
# Using old version https://github.com/meta-llama/llama/tree/57b0eb62de0636e75af471e49e2f1862d908d9d8

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    # For debugging purposses in the vscode, trying with 7b one gpu
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    # Variables set by torch.distributed.init_process_group("nccl")
    # for two gpu requirements like =< 13b, two process will be initiated
    # one with rank 0 and one with rank1 where world size is 2
    # For one gpu settings like 7b, only one process will be initiated
    # rank 0 and worldsize 1
    print(f'{os.environ["RANK"]=}')
    print(f'{os.environ["WORLD_SIZE"]=}')
    print(f'{os.environ["MASTER_ADDR"]=}')
    print(f'{os.environ["MASTER_PORT"]=}')

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    print(f"{local_rank=}")
    print(f"{world_size=}")

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def get_llama_generator(
    ckpt_dir: str = os.path.join(os.getenv("MYHOME"), "gits/llama/llama_weights/7B"),
    tokenizer_path: str = os.path.join(
        os.getenv("MYHOME"), "gits/llama/llama_weights/tokenizer.model"
    ),
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 2000,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    return load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )


def get_llama_tokenizer(
    tokenizer_path: str = os.path.join(
        os.getenv("MYHOME"), "gits/llama/llama_weights/tokenizer.model"
    ),
):
    return Tokenizer(model_path=tokenizer_path)


def generate_at_a_time(
    prompts,
    ckpt_dir: str = os.path.join(os.getenv("MYHOME"), "gits/llama/llama_weights/7B"),
    tokenizer_path: str = os.path.join(
        os.getenv("MYHOME"), "gits/llama/llama_weights/tokenizer.model"
    ),
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 2000,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    return generator.generate(
        prompts, max_gen_len=max_seq_len, temperature=temperature, top_p=top_p
    )


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 256,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
        Sentiment: Negative
        ###
        Tweet: "My day has been 👍"
        Sentiment: Positive
        ###
        Tweet: "This is the link to the article"
        Sentiment: Neutral
        ###
        Tweet: "This new music video was incredibile"
        Sentiment:""",
        """Translate English to French:
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]

    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
