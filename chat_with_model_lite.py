import argparse
import torch
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description='LLM Chat Interface')
    parser.add_argument('--model', type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help='模型路径或名称')
    parser.add_argument('--prompt', type=str, required=True,
                        help='用户提示文本')
    parser.add_argument('--system', type=str, default="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
                        help='系统提示文本')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='最大生成token数量')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-K采样参数')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-P采样参数')
    parser.add_argument('--do_sample', action='store_true',
                        help='是否使用采样')
    parser.add_argument('--no_sample', action='store_false', dest='do_sample',
                        help='是否禁用采样')
    parser.set_defaults(do_sample=True)

    args = parser.parse_args()

    # 加载模型和分词器
    pipe = pipeline("text-generation",
                    model=args.model,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto")

    # 构建消息模板
    messages = [
        {
            "role": "system",
            "content": args.system
        },
        {
            "role": "user",
            "content": args.prompt
        }
    ]

    # 应用聊天模板
    try:
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f"警告: 无法应用聊天模板，使用原始格式: {e}")
        prompt = f"### System\n{args.system}\n### User\n{args.prompt}\n### Assistant:"

    # 生成回复
    outputs = pipe(prompt,
                  max_new_tokens=args.max_tokens,
                  do_sample=args.do_sample,
                  temperature=args.temperature,
                  top_k=args.top_k,
                  top_p=args.top_p)

    # 打印结果
    print(outputs[0]["generated_text"])

if __name__ == "__main__":
    main()