# Modified by Charlie Fang from https://github.com/ml-explore/mlx-examples/blob/main/lora/lora.py
#
# Copyright © 2023 Apple Inc.
import math
import time
from typing import Union
from pathlib import Path
from data.data_utils import load_datasets, build_parser
import matplotlib.pyplot as plt

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from utils import get_model_and_tokenizer


"""
Example command for supervised fine-tuning with soft-prompts on generated data with a locally saved tiny llama:
python sft.py --prompt-tuning --save-file prompt_weights.npz --data-base increasing_mult_2_ --model ../tiny_llama --train

Step 1: Creating a model that is already prepared to do digit generation and then needs to be fine-tuned for even digits
python sft.py --save-file digit_fine_tune.npz --data-base increasing_mult_1_ --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --train --iters 20000 --data ./data/ --batch-size 32 --steps-per-eval 21000
Step 2: Fine tuning to prepare for even-digit generation
python sft.py --save-file even_digit_fine_tune.npz --data-base increasing_mult_2_ --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --resume-file digit_fine_tune.npz --train --iters 50 --data ./data/ --batch-size 32
Step 3: Learning a reward model from preference data
python pytorch_sft.py --reward-model --train --data-base reward_function_increasing_mult_2_  --save-file even_reward_model --model ./digit_fine_tune/ --iters 20000 --data ../data/ --batch-size 32 --steps-per-eval 21000



Example command for training a reward model with LoRA on generated data with a HF tiny llama
python sft.py --reward-model --train --data-base reward_function_increasing_mult_2_ --batch-size 16 --save-file reward_lora.npz --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""


def loss(mdl, inputs, targets, input_lengths, target_lengths=None):
    """
    SFT loss, standard language modeling cross-entropy loss
    支持input和target长度不一致的情况

    参数:
        mdl: 模型
        inputs: 输入序列
        targets: 目标序列
        input_lengths: 输入序列的有效长度
        target_lengths: 目标序列的有效长度(可选，默认为input_lengths)
    """
    # 如果未提供目标长度，则默认为输入长度
    if target_lengths is None:
        target_lengths = input_lengths

    # 运行模型获取logits
    logits = mdl(inputs)

    # 确保logits和targets的维度匹配
    if logits.shape[1] > targets.shape[1]:
        # 截断logits到targets的长度
        logits = logits[:, :targets.shape[1], :]
    elif logits.shape[1] < targets.shape[1]:
        # 这种情况通常不应该发生，因为模型输出应该至少和输入一样长
        raise ValueError(f"Logits长度({logits.shape[1]})小于targets长度({targets.shape[1]})")

    # 创建目标序列的掩码
    # 注意：这里使用target_lengths而不是input_lengths
    target_mask = mx.arange(targets.shape[1])[None, :] < target_lengths[:, None]

    # 处理特殊的填充标记(-100)
    if -100 in targets[0]:
        # 将标记为-100的位置从掩码中排除
        target_mask = target_mask & (targets != -100)

    # 计算交叉熵损失
    # 注意：logits和targets现在可能具有不同的长度，但我们已经截断了logits
    ce = nn.losses.cross_entropy(logits, targets) * target_mask

    # 计算有效token数
    ntoks = target_mask.sum()

    # 计算平均损失
    if ntoks == 0:
        return mx.array(0.0), ntoks  # 避免除零错误
    else:
        ce = ce.sum() / ntoks

    return ce, ntoks


def reward_loss(mdl, better_inputs, worse_inputs):
    """
    Reward modeling loss, maximizing the difference between the preferred sequence and the "dispreferred" sequence
    (Assumes that the reward for seq1 >= reward for seq2)
    Returns:
        Loss value, tokens-per-second (TODO -- Tokens-per-second implementation missing here)
    """
    # TODO: Batch these, currently this is unnecessarily slow.
    _, _, rewards_j = mdl(better_inputs)
    _, _, rewards_k = mdl(worse_inputs)
    # Batch x SeqLen x OutputDim -- get last token value
    diff_val = -mx.log(mx.sigmoid(rewards_j[:, -1, :] - rewards_k[:, -1, :])).mean()
    return diff_val, mx.array(0)  # TODO: this is telling the logger "0 toks per sec"


def iterate_batches(dset, tok, batch_size, train_mode=False, reward_modeling=False, chat_data=False):
    # Shuffle indices
    len_warning_message = "[WARNING] Some sequences are longer than 2048 tokens. " \
                          "Consider pre-splitting your data to save memory."
    while True:
        indices = np.arange(len(dset))
        if train_mode:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            if reward_modeling:
                pref_batch, bad_batch = [], []
                p_lengths, b_lengths = [], []
                for j in range(batch_size):
                    pref, bad = dset[indices[i + j]]
                    pref_batch.append(tok.encode(pref))
                    p_lengths.append(len(pref_batch[-1]))
                    bad_batch.append(tok.encode(bad))
                    b_lengths.append(len(bad_batch[-1]))
                if max(max(p_lengths), max(b_lengths)) > 2048:
                    print(len_warning_message)
                p_arr = np.zeros((batch_size, max(p_lengths)), np.int32)
                b_arr = np.zeros((batch_size, max(b_lengths)), np.int32)
                for j in range(batch_size):
                    p_arr[j, : p_lengths[j]] = pref_batch[j]
                    b_arr[j, : b_lengths[j]] = bad_batch[j]
                pref_batch = mx.array(p_arr)
                bad_batch = mx.array(b_arr)
                yield pref_batch, bad_batch
            else:
                if chat_data:
                    batch = [dset[indices[i + j]] for j in range(batch_size)]
                    input_ids = [x['input_ids'] for x in batch]
                    labels = [x['labels'] for x in batch]
                    lengths = [len(x['input_ids']) for x in batch]
                    batch_arr = np.ones((batch_size, max(lengths)), np.int32) * tok.pad_token_id
                    label_arr = np.ones_like(batch_arr) * -100
                    for j in range(batch_size):
                        batch_arr[j, : lengths[j]] = input_ids[j]
                        label_arr[j, : lengths[j]] = labels[j]
                    batch = mx.array(batch_arr)
                    targets = mx.array(label_arr)
                else:
                    samples = [dset[indices[i + j]] for j in range(batch_size)]

                    # 分离输入和目标
                    if dset.standard_sft:
                        inputs = [sample[0] for sample in samples]
                        targets = [sample[1] for sample in samples]
                    else:
                        inputs = samples
                        targets = samples

                    # 批量编码（利用fast tokenizer的并行能力）
                    encoding = tok(inputs, padding="longest", truncation=True, max_length=2048, return_tensors="np")
                    batch = mx.array(encoding["input_ids"])
                    lengths = [len(x) for x in batch]

                    # 对目标文本进行同样的批量处理
                    target_encoding = tok(targets, padding="longest", truncation=True, max_length=2048,
                                          return_tensors="np")
                    targets = mx.array(target_encoding["input_ids"])

                    # 检查超长序列
                    if encoding["input_ids"].shape[1] > 2048 or target_encoding["input_ids"].shape[1] > 2048:
                        print(len_warning_message)
                yield batch[:, :-1], targets[:, 1:], mx.array(lengths)

        if not train_mode:
            break


def evaluate(mdl, dataset, loss_fn, tok, train_args):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
            range(train_args.val_batches),
            iterate_batches(dataset, tok, train_args.batch_size,
                            reward_modeling=train_args.reward_model, chat_data=train_args.data_base == 'chat'),
    ):
        losses, toks = loss_fn(mdl, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / max(ntokens, train_args.val_batches)


def save_adapter(
    save_model: nn.Module,
    adapter_file: Union[str, Path],
):
    flattened_tree = tree_flatten(save_model.trainable_parameters())
    mx.save_safetensors(str(adapter_file), dict(flattened_tree))


def train(mdl, train_ds, val_set, optimizer, loss_fn, tok, train_args):
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(mdl, loss_fn)

    losses = []
    val_losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
            range(train_args.iters),
            iterate_batches(train_ds, tok, train_args.batch_size,
                            train_mode=True, reward_modeling=train_args.reward_model,
                            chat_data=train_args.data_base == 'chat'),
    ):

        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(mdl, *batch)

        # Model update
        optimizer.update(mdl, grad)
        mx.eval(mdl.parameters(), optimizer.state, lvalue)

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks.item()

        # Report training loss if needed
        if (it + 1) % train_args.steps_per_report == 0:
            train_loss = np.mean(losses[-train_args.steps_per_report:])

            stop = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {train_args.steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if (it == 0 or (it + 1) % train_args.steps_per_eval == 0) and val_set is not None:
            stop = time.perf_counter()
            val_loss = evaluate(
                mdl, val_set, loss_fn, tok, train_args
            )
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )
            val_losses.append(val_loss)

            start = time.perf_counter()

        # Save prompt weights if needed
        if (it + 1) % train_args.save_every == 0:
            save_adapter(model, train_args.save_file)
            checkpoint = (
                    Path(train_args.save_file).parent / f"{it:07d}_adapters.safetensors"
            )
            save_adapter(model, checkpoint)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{train_args.save_file} and {checkpoint}."
            )
    fn = ''
    if train_args.prompt_tuning:
        fn += 'prompt_tuning_'
    else:
        fn += 'lora_'
    plt.plot(losses)
    plt.savefig(f'{fn}train_losses.png')
    plt.plot(val_losses)
    plt.savefig(f'{fn}val_losses.png')


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    model, tokenizer = get_model_and_tokenizer(args)

    print("Loading datasets")
    train_set, valid_set, test_set = load_datasets(args, tokenizer)

    if args.reward_model:
        loss_function = reward_loss
    else:
        loss_function = loss

    if args.train:
        print("Training")
        opt = optim.Adam(learning_rate=args.learning_rate)

        # Train model
        train(model, train_set, valid_set, opt, loss_function, tokenizer, args)

        # Save weights
        mx.savez(args.save_file, **dict(tree_flatten(model.trainable_parameters())))

    # Load the weights which we assume should exist by this point
    if not Path(args.save_file).is_file():
        raise ValueError(
            f"Save file {args.save_file} missing. "
            "Use --train to learn and save the prompts.npz."
        )
    model.load_weights(args.save_file, strict=False)

    if args.test and test_set is not None:
        print("Testing")
        model.eval()
        test_loss = evaluate(
            model,
            test_set,
            loss,
            tokenizer,
            args
        )
        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
