import os
import sys
import torch
import logging
from datasets import load_dataset
import datasets
import transformers
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# 设置设备
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 加载模型和分词器
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充标记

# 加载数据集
data_files = {"train": "data/alpaca_2000.jsonl"}
dataset = load_dataset("json", data_files=data_files)


# 预处理数据集
def preprocess_function(examples):
    inputs, targets = (example for example in examples.values())
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 配置LoRA参数
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # LoRA矩阵的秩
    lora_alpha=32,  # LoRA缩放因子
    lora_dropout=0.1,  # Dropout概率
    bias="none",  # 不训练偏置项
    target_modules=["q_proj", "v_proj"]  # 只对查询和值投影矩阵应用LoRA
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# 应用LoRA配置
model = get_peft_model(model, peft_config)

# 打印可训练参数信息
model.print_trainable_parameters()

# 定义训练参数
training_args = TrainingArguments(
    output_dir="/Users/aequatio/147-大模型本地/results_alpaca",
    learning_rate=2e-5,
    optim="adamw_torch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=8,
    weight_decay=0.01,
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    logging_dir="./logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="wandb",
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

# 创建数据收集器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"] if "valid" in tokenized_datasets else None,
    data_collator=data_collator,
)
wandb.init(
    project="RLAIF",
    name="test-run-alpaca-fix-data",
    tags=["baseline"],
    group="llama",
)
# 开始训练
trainer.train()

# 保存微调后的模型
model.save_pretrained("/Users/aequatio/147-大模型本地/tinyllama-alpaca-fix")
tokenizer.save_pretrained("/Users/aequatio/147-大模型本地/tinyllama-alpaca-fix")

# 保存LoRA权重（更小的文件）
model.base_model.save_pretrained("/Users/aequatio/147-大模型本地/tinyllama-alpaca-fix")