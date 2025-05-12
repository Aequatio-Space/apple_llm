import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
# pipe = pipeline("text-generation", model="andrewsilva/increasing_even_digit_fine_tune", torch_dtype=torch.bfloat16, device_map="auto")
# pipe = pipeline("text-generation", model="/Users/aequatio/147-大模型本地/sft_fine_tune_batch_token", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "Summarize the following paragraph with a topic and a few keywords.",
    },
    {"role": "user", "content": "Introduction To Swift Programming : Swift Cheat Sheet - A comprehensive Guide on Classes, Protocols, Extensions, Enums, Initializers, and Deinitializers"},
]
# messages = [
#     {
#         "role": "system",
#         "content": "You're a helpful assistant.",
#     },
#     {"role": "user", "content": "16 26 72 104 152 172 184"},
# ]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])