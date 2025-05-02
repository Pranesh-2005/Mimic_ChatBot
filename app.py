import os
import re
import json
import torch
import gradio as gr
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s?[ap]m) - ([^:]+): (.*)$")

def process_txt_file(file_path):
    pairs = []
    last_user, last_bot = [], []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = pattern.match(line.strip())
            if not match: continue
            _, _, sender, message = match.groups()

            if sender.lower() == "you":
                if last_bot:
                    pairs.append({"input": " ".join(last_user), "output": " ".join(last_bot)})
                    last_bot = []
                last_user.append(message.strip())
            elif sender.lower() in ["bot", "🦋 حب"]:
                last_bot.append(message.strip())

    if last_user and last_bot:
        pairs.append({"input": " ".join(last_user), "output": " ".join(last_bot)})
    return pairs

def fine_tune_model(pairs):
    dataset = Dataset.from_dict({
        "input": [p["input"] for p in pairs],
        "output": [p["output"] for p in pairs]
    })

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    def tokenize(example):
        inputs = tokenizer(example["input"], truncation=True, max_length=128, padding="max_length")
        outputs = tokenizer(example["output"], truncation=True, max_length=128, padding="max_length")
        inputs["labels"] = outputs["input_ids"]
        return inputs

    tokenized = dataset.map(tokenize, remove_columns=["input", "output"])

    training_args = TrainingArguments(
        output_dir="./chatbot_model",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_strategy="epoch",
        logging_dir="./logs",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    model.save_pretrained("./chatbot_model")
    tokenizer.save_pretrained("./chatbot_model")
    return model, tokenizer

model, tokenizer = None, None

def train_and_chat(file):
    global model, tokenizer
    file_path = file.name
    pairs = process_txt_file(file_path)
    if len(pairs) < 2:
        return "❌ Not enough chat data to train."
    model, tokenizer = fine_tune_model(pairs)
    return "✅ Training complete! Start chatting..."

def chat_response(user_input):
    if model is None or tokenizer is None:
        return "Please upload a chat file and train the model first."

    prompt = f"You: {user_input}\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Bot:" in response:
        return response.split("Bot:")[-1].strip()
    return response.strip()

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Dynamic GPT-2 WhatsApp Chatbot")
    upload = gr.File(label="Upload WhatsApp Chat (.txt)")
    status = gr.Textbox(label="Status", interactive=False)
    upload_button = gr.Button("Train Model")

    upload_button.click(fn=train_and_chat, inputs=upload, outputs=status)

    input_text = gr.Textbox(lines=2, label="You")
    output_text = gr.Textbox(label="Bot")
    submit = gr.Button("Send")

    submit.click(fn=chat_response, inputs=input_text, outputs=output_text)

demo.launch()
