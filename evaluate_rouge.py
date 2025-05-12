import json
import argparse
import numpy as np
from rouge import Rouge
from tqdm import tqdm
from transformers import pipeline
import torch


def main(args):
    # 初始化pipeline
    pipe = pipeline(
        "text-generation",
        model=args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    # 初始化ROUGE评估器
    rouge = Rouge()

    # 存储所有预测和参考文本
    predictions = []
    references = []

    # 从jsonl文件读取数据
    print(f"Loading data from {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"Evaluating {len(data)} samples...")

    # 生成预测并收集结果
    for example in tqdm(data[:10]):
        input_text = example["text"]
        target_text = example["text"]
        if args.add_template:
            messages = [
                {
                    "role": "system",
                    "content": "You're a helpful assistant.",
                },
                {"role": "user", "content": input_text},
            ]
            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = input_text

        # 生成模型预测
        outputs = pipe(
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=1
        )

        # 提取生成的文本（去除输入部分）
        generated_text = outputs[0]["generated_text"][len(prompt):].strip()

        # 存储预测和参考文本
        predictions.append(generated_text)
        references.append(target_text)

    # 计算ROUGE分数
    try:
        scores = rouge.get_scores(predictions, references, avg=True)
        print("\nROUGE Scores:")
        print(f"ROUGE-1: {scores['rouge-1']['f']:.4f} (p={scores['rouge-1']['p']:.4f}, r={scores['rouge-1']['r']:.4f})")
        print(f"ROUGE-2: {scores['rouge-2']['f']:.4f} (p={scores['rouge-2']['p']:.4f}, r={scores['rouge-2']['r']:.4f})")
        print(f"ROUGE-L: {scores['rouge-l']['f']:.4f} (p={scores['rouge-l']['p']:.4f}, r={scores['rouge-l']['r']:.4f})")

        # 保存结果
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(scores, f, indent=4)
            print(f"Scores saved to {args.output_file}")

            # 保存详细结果
            if args.save_details:
                details_file = args.output_file.replace('.json', '_details.json')
                with open(details_file, 'w', encoding='utf-8') as f:
                    json.dump([
                        {"prediction": pred, "reference": ref}
                        for pred, ref in zip(predictions, references)
                    ], f, indent=4, ensure_ascii=False)
                print(f"Details saved to {details_file}")

    except Exception as e:
        print(f"Error calculating ROUGE scores: {e}")
        scores = None

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model ROUGE scores on a dataset")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save scores (JSON)")
    parser.add_argument("--save_details", action="store_true", help="Save detailed predictions and references")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code from HF")
    parser.add_argument('--add-template', action='store_true', help='Add template to the input text')

    args = parser.parse_args()
    main(args)