# Dynamic GPT-2 WhatsApp Chatbot

Train a personalized chatbot by uploading your own WhatsApp `.txt` chat exports. The app fine-tunes GPT-2 based on your real messages and launches an interactive chatbot for chatting!

## Features
- Upload your chat logs in upload folder
- Dowload the gpt2 model and place it in model
- Real-time Gradio chat interface
- Supports emojis and personalized style

## Setup
- Upload your chat logs in upload folder
- Dowload the gpt2 model and place it in model
- Also in the app.py file change the line no 28,33 (if sender.lower() == "you":) and elif sender.lower() in ["bot", "🦋 حب"]: to your and the person you want to mimic and replace it by the names present in WhatsApp .txt file

```bash
pip install -r requirements.txt
python app.py
