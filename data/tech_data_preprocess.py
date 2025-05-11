import pyarrow.parquet as pq
import json
import argparse
from pathlib import Path
import pandas as pd


def process_parquet_to_jsonl(input_path, output_path=None, batch_size=1000):
    """
    从Parquet文件中提取指定字段并保存为JSONL格式

    参数:
    input_path (str): 输入Parquet文件路径
    output_path (str, optional): 输出JSONL文件路径，默认为None(自动生成)
    batch_size (int): 每次处理的行数，用于内存优化
    """
    # 确保输入文件存在
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 自动生成输出路径(如果未指定)
    if output_path is None:
        output_path = input_file.with_name(f"{input_file.stem}.jsonl")

    # 定义要提取的字段
    selected_fields = ["text", "keywords", "topic"]

    try:
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
                    # 创建输出记录
                    record = {
                        "text": str(row["text"]),  # 确保text字段为字符串类型
                        "keywords": row["keywords"],
                        "topic": row["topic"]
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
    parser = argparse.ArgumentParser(description='将Parquet文件转换为JSONL格式')
    parser.add_argument('--input', required=True, help='输入Parquet文件路径')
    parser.add_argument('--output', help='输出JSONL文件路径(可选)')
    parser.add_argument('--batch-size', type=int, default=1000, help='每次处理的行数(默认1000)')

    # 解析命令行参数
    args = parser.parse_args()

    # 执行转换
    process_parquet_to_jsonl(args.input, args.output, args.batch_size)