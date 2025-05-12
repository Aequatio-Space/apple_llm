# Modified by Charlie Fang from https://github.com/ml-explore/mlx-examples/blob/main/lora/lora.py
#
# Copyright © 2023 Apple Inc.
import math
import time
from typing import Union
import logging
from datetime import datetime
from pathlib import Path
from data.data_utils import load_datasets, build_parser
from rouge import Rouge
import nltk
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
    logits, _, _ = mdl(inputs)

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
                            reward_modeling=train_args.reward_model,
                            chat_data=train_args.data_base == 'chat'),
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
    # 创建日志记录器
    log_file, current_date = setup_logger(train_args)

    # 确保nltk数据已下载
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # 创建损失函数的值和梯度计算函数
    loss_value_and_grad = nn.value_and_grad(mdl, loss_fn)

    # 初始化记录器
    losses = []  # 训练损失
    val_losses = []  # 验证损失
    rouge_scores = []  # ROUGE分数
    n_tokens = 0

    # 主训练循环
    start = time.perf_counter()
    for it, batch in zip(
            range(train_args.iters),
            iterate_batches(train_ds, tok, train_args.batch_size,
                            train_mode=True, reward_modeling=train_args.reward_model,
                            chat_data=train_args.data_base == 'chat'),
    ):

        # 前向和反向传播
        (lvalue, toks), grad = loss_value_and_grad(mdl, *batch)

        # 模型更新
        optimizer.update(mdl, grad)
        mx.eval(mdl.parameters(), optimizer.state, lvalue)

        # 记录损失
        losses.append(lvalue.item())
        n_tokens += toks.item()

        # 定期报告训练损失
        if (it + 1) % train_args.steps_per_report == 0:
            train_loss = np.mean(losses[-train_args.steps_per_report:])

            stop = time.perf_counter()
            log_message = (
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {train_args.steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
            logging.info(log_message)
            n_tokens = 0
            start = time.perf_counter()

        # 定期进行验证并计算评估指标
        if (it == 0 or (it + 1) % train_args.steps_per_eval == 0) and val_set is not None:
            stop = time.perf_counter()

            # 计算验证损失
            val_loss = evaluate(
                mdl, val_set, loss_fn, tok, train_args
            )
            val_losses.append(val_loss)

            # 计算ROUGE和BLEU分数
            if train_args.calculate_metrics:
                rouge_score = evaluate_metrics(
                    mdl, val_set, tok, train_args
                )
                rouge_scores.append(rouge_score)

                # 记录评估结果
                log_message = (
                    f"Iter {it + 1}: "
                    f"Val loss {val_loss:.3f}, "
                    f"ROUGE-L: {rouge_score['rouge-l']['f']:.4f}, "
                    f"Val took {(time.perf_counter() - stop):.3f}s"
                )
                logging.info(log_message)
            else:
                log_message = (
                    f"Iter {it + 1}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {(time.perf_counter() - stop):.3f}s"
                )
                logging.info(log_message)

            start = time.perf_counter()

        # 定期保存模型
        if (it + 1) % train_args.save_every == 0:
            save_adapter(model, train_args.save_file)
            checkpoint = (
                    Path(train_args.save_file).parent / f"{it:07d}_adapters.safetensors"
            )
            save_adapter(model, checkpoint)
            log_message = (
                f"Iter {it}: Saved adapter weights to "
                f"{train_args.save_file} and {checkpoint}."
            )
            logging.info(log_message)

    # 确定文件名前缀
    fn = ''
    if train_args.prompt_tuning:
        fn += 'prompt_tuning_'
    else:
        fn += 'lora_'

    # 绘制并保存所有指标曲线
    plot_and_save_metrics(
        losses, val_losses, rouge_scores,
        f'{fn}training_metrics_{current_date}.png', train_args
    )

    logging.info(f"训练完成，日志保存到: {log_file}")


def setup_logger(train_args) -> tuple:
    """配置日志记录器，将日志输出到文件和控制台"""
    # 获取当前日期
    current_date = datetime.now().strftime("%Y%m%d-%H%M")

    # 从save_file获取基础名称
    base_name = Path(train_args.save_file).stem

    # 构建日志文件名: 日期+基础名称+".log"
    log_dir = Path(train_args.save_file).parent
    log_file = log_dir / f"{current_date}_{base_name}.log"

    # 创建日志目录(如果不存在)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # 记录训练配置
    logging.info(f"训练配置: {vars(train_args)}")
    return log_file, current_date


def generate_text(model, inputs, max_length=512, temp=0.0):
    generated_tokens = []
    for token in model.generate(inputs, temp):
        generated_tokens.append(token)
        if len(generated_tokens) >= max_length:
            break
    return generated_tokens


def evaluate_metrics(mdl, val_set, tok, train_args):
    """计算验证集上的ROUGE和BLEU指标"""
    references = []
    predictions = []

    # 生成预测结果
    for batch in iterate_batches(val_set, tok, train_args.batch_size,
                                 train_mode=False, reward_modeling=train_args.reward_model,
                                 chat_data=train_args.data_base == 'chat'):
        inputs, targets = batch[:2]  # 获取输入和目标

        # 生成模型预测（使用改进的生成函数）
        generated = generate_text(mdl, inputs,
                                  max_length=train_args.max_gen_length,
                                  temp=train_args.temp)

        # 批量解码生成的文本
        generated_np = np.array([gen_tokens for gen_tokens in generated]).T
        pred_texts = tok.batch_decode(generated_np, skip_special_tokens=True)

        # 批量解码目标文本
        targets_np = [np.array(t) for t in targets]
        target_texts = tok.batch_decode(targets_np, skip_special_tokens=True)

        # 确保预测和参考长度一致
        if len(pred_texts) != len(target_texts):
            min_len = min(len(pred_texts), len(target_texts))
            pred_texts = pred_texts[:min_len]
            target_texts = target_texts[:min_len]

        references.extend(target_texts)
        predictions.extend(pred_texts)

    # 计算ROUGE分数
    # print shape of first item in predictions and references
    logging.info(f"Predictions sample: {predictions[0]}")
    logging.info(f"References sample: {references[0]}")
    rouge = Rouge()
    try:
        rouge_score = rouge.get_scores(predictions, references, avg=True)
    except Exception as e:
        logging.warning(f"ROUGE计算出错: {e}")
        rouge_score = {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
                       'rouge-2': {'f': 0, 'p': 0, 'r': 0},
                       'rouge-l': {'f': 0, 'p': 0, 'r': 0}}

    return rouge_score


def plot_and_save_metrics(losses, val_losses, rouge_scores, filename, train_args):
    """绘制并保存所有训练指标"""
    plt.figure(figsize=(15, 10))

    # 绘制训练损失
    plt.subplot(2, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    x_vals = [i for i in range(0, train_args.iters + train_args.steps_per_eval,
                               train_args.steps_per_eval)]

    # 绘制验证损失
    if val_losses:
        plt.subplot(2, 2, 2)

        if len(x_vals) > len(val_losses):
            x_vals = x_vals[:len(val_losses)]
        plt.plot(x_vals, val_losses)
        plt.title('Validation Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

    # 绘制ROUGE分数
    if rouge_scores:
        plt.subplot(2, 2, 3)
        if len(x_vals) > len(rouge_scores):
            x_vals = x_vals[:len(rouge_scores)]

        # 提取ROUGE-L的F1分数
        rouge_l_f1 = [score['rouge-l']['f'] for score in rouge_scores]
        plt.plot(x_vals, rouge_l_f1)
        plt.title('ROUGE-L F1 Score')
        plt.xlabel('Iterations')
        plt.ylabel('Score')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Metrics plot saved to {filename}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    model, tokenizer = get_model_and_tokenizer(args, need_generate=True)

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
        if args.calculate_metrics:
            rouge_score = evaluate_metrics(
                model,
                test_set,
                tokenizer,
                args
            )
            print(f"Test ROUGE-L: {rouge_score['rouge-l']['f']:.4f}")
        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
