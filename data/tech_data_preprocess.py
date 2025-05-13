import pyarrow.parquet as pq
import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer


def process_parquet_to_jsonl(input_path, output_path=None, batch_size=1000,
                             model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                             system_prompt="You will receive a text. Your task is to extract keywords and topic from it."):
    """
    从Parquet文件中提取指定字段并保存为JSONL格式，同时构建对话模板

    参数:
    input_path (str): 输入Parquet文件路径
    output_path (str, optional): 输出JSONL文件路径，默认为None(自动生成)
    batch_size (int): 每次处理的行数，用于内存优化
    model_name (str): 用于构建对话模板的模型名称
    """
    # 确保输入文件存在
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 自动生成输出路径(如果未指定)
    if output_path is None:
        output_path = input_file.with_name(f"{input_file.stem}.jsonl")

    # 定义要提取的字段
    selected_fields = ['instruction', 'output', 'input']

    try:
        # 加载tokenizer用于构建对话模板
        print(f"加载tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 打开Parquet文件
        parquet_file = pq.ParquetFile(input_path)

        # 获取文件总记录数
        total_rows = parquet_file.metadata.num_rows
        print(f"开始处理Parquet文件，总记录数: {total_rows}")

        # 分批次读取数据
        with open(output_path, 'w', encoding='utf-8') as f:
            for batch_idx, batch in enumerate(
                    parquet_file.iter_batches(batch_size=batch_size, columns=selected_fields)):
                # 转换为Pandas DataFrame
                df = batch.to_pandas()

                # 检查是否所有字段都存在
                missing_fields = [field for field in selected_fields if field not in df.columns]
                if missing_fields:
                    raise ValueError(f"Parquet文件中缺少以下字段: {', '.join(missing_fields)}")

                # 处理每一行并写入JSONL
                for _, row in df.iterrows():
                    # 构建对话模板
                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": str(row["instruction"]) + f'\\<|input|>{str(row["input"])}</s>'  # 确保text为字符串类型
                        },
                    ]
                    # 使用tokenizer的对话模板功能
                    try:
                        # 构建包含特殊标记的输入文本
                        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    except Exception as e:
                        # 如果tokenizer不支持apply_chat_template，手动构建
                        print(f"警告: tokenizer不支持apply_chat_template，使用手动构建: {e}")
                        input_text = f"<|system|>\n{messages[0]['content']}</s><|user|>\n{messages[1]['content']}</s>"
                        if len(messages[2]['content']) > 0:
                            input_text += '<|input|>\n' + messages[2]['content'] + '</s>'
                        input_text += "<|assistant|>"

                    # 构建目标输出: "Keywords: XXX, Topic: XXX"
                    target_text = row['output']
                    # 创建输出记录
                    record = {
                        "input": input_text,
                        "target": target_text
                    }

                    # 写入JSONL格式
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

                # 打印进度
                if (batch_idx + 1) % 10 == 0:
                    processed_rows = (batch_idx + 1) * batch_size
                    progress = min(processed_rows / total_rows * 100, 100)
                    print(f"已处理: {processed_rows}/{total_rows} 行 ({progress:.2f}%)")

        print(f"处理完成，输出文件保存在: {output_path}")
        return output_path

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将Parquet文件转换为JSONL格式，添加对话模板')
    parser.add_argument('--input', required=True, help='输入Parquet文件路径')
    parser.add_argument('--output', help='输出JSONL文件路径(可选)')
    parser.add_argument('--batch-size', type=int, default=1000, help='每次处理的行数(默认1000)')
    parser.add_argument('--model', default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help='用于对话模板的模型名称')

    # 解析命令行参数
    args = parser.parse_args()

    # 执行转换
    process_parquet_to_jsonl(args.input, args.output, args.batch_size,
                             args.model, "Below is an instruction that describes a task. Write a response that appropriately completes the request.")