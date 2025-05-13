from rouge import Rouge

# 模拟预测文本和参考文本
# 注意：predictions和references都应为字符串列表
predictions = [
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is revolutionizing the tech industry",
    "Machine learning models require大量的训练数据"
]

references = [
    "The fast brown fox leaps over the sleepy dog",  # 与第一个预测相似但不完全相同
    "AI is transforming the technology sector",  # 与第二个预测同义但表述不同
    "机器学习模型需要大量的训练数据"  # 与第三个预测完全相同（中文示例）
]

# 初始化ROUGE评估器
rouge = Rouge()

try:
    # 计算ROUGE分数（avg=True表示返回平均值）
    rouge_score = rouge.get_scores(predictions, references, avg=True)

    # 打印结果
    print("ROUGE分数计算结果:")
    for metric, values in rouge_score.items():
        print(f"{metric}:")
        print(f"  F1-score: {values['f']:.4f}")
        print(f"  Precision: {values['p']:.4f}")
        print(f"  Recall: {values['r']:.4f}")
        print()

except Exception as e:
    print(f"ROUGE计算出错: {e}")
    rouge_score = {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
                   'rouge-2': {'f': 0, 'p': 0, 'r': 0},
                   'rouge-l': {'f': 0, 'p': 0, 'r': 0}}