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


def loss(mdl, inputs, targets, lengths):
    """
    SFT loss, standard language modeling cross-entropy loss
    Returns:
        Loss value, tokens-per-second
    """
    # Run model on inputs
    logits = mdl(inputs)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    if -100 in targets[0]:  # If we are masking some targets
        # Cast to numpy because mlx doesn't support boolean indexing
        np_len = np.array(length_mask)
        # Mask out targets
        np_len[targets == -100] = False
        # Cast back to mlx
        length_mask = mx.array(np_len)

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
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


def generate_text(model, inputs, tokenizer, max_length=512, temp=1.0,
                  top_k=40, top_p=0.95, eos_token_id=None):
    """
    使用更高级的采样策略生成文本，支持批处理

    参数:
        model: 训练好的模型
        inputs: 输入张量，形状为 [batch_size, seq_len]
        tokenizer: 用于解码的tokenizer
        max_length: 最大生成长度
        temp: 温度参数，控制随机性
        top_k: top-k采样的k值
        top_p: top-p采样的p值
        eos_token_id: 结束标记的token ID
    """
    # 获取结束标记ID（如果未指定，则从tokenizer获取）
    if eos_token_id is None and hasattr(tokenizer, 'eos_token_id'):
        eos_token_id = tokenizer.eos_token_id

    batch_size = inputs.shape[0]
    generated = inputs
    eos_found = [False] * batch_size  # 跟踪每个样本是否已生成EOS

    for _ in range(max_length):
         # 模型推理，获取logits、新的cache和value
        logits, _, _ = model(generated)  # 只需要logits
        logits = logits[:, -1, :]  # 获取最后一个位置的logits

        # 采样策略
        if temp > 0:
            # 应用温度缩放
            logits = logits / temp

            # Top-k采样
            if top_k > 0:
                # MLX的sort默认升序，所以取负号实现降序
                sorted_logits = mx.sort(-logits, axis=-1)  # 注意这里取负号
                sorted_logits = -sorted_logits  # 恢复原始值
                kth_value = sorted_logits[:, top_k - 1:top_k]  # 获取第k大的值
                indices_to_remove = logits < kth_value
                logits = logits - 1e10 * indices_to_remove  # 将低于阈值的logits设为负无穷

            # Top-p采样（核采样）
            if top_p < 1.0:
                # MLX的sort默认升序，所以取负号实现降序
                sorted_logits = mx.sort(-logits, axis=-1)
                sorted_logits = -sorted_logits  # 恢复原始值
                sorted_indices = mx.argsort(-logits, axis=-1)  # 注意这里取负号

                cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

                # 移除累积概率超过阈值的token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个token（即使超过阈值）
                sorted_indices_to_remove = mx.concatenate([
                    mx.zeros_like(sorted_indices_to_remove[..., :1]),
                    sorted_indices_to_remove[..., 1:]
                ], axis=-1)

                # 将被移除的token的logits设为负无穷
                indices_to_remove = mx.zeros_like(logits)
                # 为每个样本手动应用掩码
                for i in range(batch_size):
                    # 获取当前样本的排序索引和掩码
                    sample_indices = sorted_indices[i]
                    sample_mask = sorted_indices_to_remove[i]

                    # 将掩码应用到原始索引位置
                    for j in range(len(sample_indices)):
                        if sample_mask[j].item():
                            indices_to_remove[i, sample_indices[j]] = True

                logits = logits - 1e10 * indices_to_remove

            # 应用softmax获取概率分布
            probs = mx.softmax(logits, axis=-1)
            # 从分布中采样
            next_tokens = mx.random.categorical(probs)
        else:
            # 贪婪解码（温度为0）
            next_tokens = mx.argmax(logits, axis=-1)

        # 将采样的token添加到生成序列中
        generated = mx.concatenate([generated, next_tokens[:, None]], axis=1)

        # 检查是否生成了EOS token
        if eos_token_id is not None:
            for i in range(batch_size):
                if not eos_found[i] and next_tokens[i].item() == eos_token_id:
                    eos_found[i] = True

        # 如果所有样本都生成了EOS token，则提前结束
        if all(eos_found):
            break

    # 转换为NumPy数组并移除输入部分
    generated_np = generated.to_numpy()
    inputs_np = inputs.to_numpy()
    generated_sequences = []

    for i in range(batch_size):
        # 提取生成的部分（不包括输入）
        generated_part = generated_np[i][len(inputs_np[i]):]
        generated_sequences.append(generated_part)

    return generated_sequences


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
        generated = generate_text(mdl, inputs, tok,
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

    model, tokenizer = get_model_and_tokenizer(args, need_generate=args.calculate_metrics)

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
