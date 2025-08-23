import gradio as gr
import os, collections
import chromadb
from chromadb.utils import embedding_functions
from dataclasses import dataclass, field
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2025-01-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Chroma for embeddings
chroma_client = chromadb.Client()
embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_type="azure",
    api_version="2025-01-01-preview",
    deployment_id="text-embedding-ada-002",  # your deployment name
    model_name="text-embedding-ada-002"
)

@dataclass
class SessionState:
    chroma_db: object = None
    loaded: bool = False

def parse_whatsapp_text(text: str):
    messages = []
    for line in text.splitlines():
        if " - " in line and ":" in line:
            try:
                _, rest = line.split(" - ", 1)
                sender, msg = rest.split(":", 1)
                messages.append(msg.strip())
            except:
                continue
    return messages

def load_chats(files, state: SessionState):
    all_msgs = []
    for f in files or []:
        try:
            txt = open(f, encoding="utf-8").read()
        except:
            txt = open(f, encoding="utf-16").read()
        all_msgs.extend(parse_whatsapp_text(txt))

    if not all_msgs:
        return "No messages found.", state

    # Build one Chroma collection for all messages
    collection = chroma_client.create_collection(
        name=f"whatsapp_{id(state)}",
        embedding_function=embedder
    )
    for i, m in enumerate(all_msgs):
        collection.add(documents=[m], ids=[f"msg_{i}"])

    state.chroma_db = collection
    state.loaded = True
    return f"Loaded {len(all_msgs)} messages into memory.", state

def respond(user_input, chat_history, state: SessionState):
    if not state.loaded or not state.chroma_db:
        return chat_history, state

    results = state.chroma_db.query(query_texts=[user_input], n_results=5)
    context = "\n".join(results["documents"][0]) if results["documents"] else ""

    completion = client.chat.completions.create(
        model="gpt-4.1",  # use your Azure deployment
        messages=[
            {"role": "system", "content": "You are chatting casually based on WhatsApp history."},
            {"role": "user", "content": f"User said: {user_input}\n\nRelevant past messages:\n{context}"}
        ]
    )
    reply = completion.choices[0].message.content

    chat_history = chat_history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": reply}]
    return chat_history, state

def purge(state: SessionState):
    state.chroma_db = None
    state.loaded = False
    return None, "Purged session data.", state

with gr.Blocks() as demo:
    gr.Markdown("# 🦜 WhatsApp Chat Mimic (Azure OpenAI + RAG)")
    state = gr.State(SessionState())

    files = gr.File(label="Upload WhatsApp .txt", file_count="multiple", type="filepath")
    load_btn = gr.Button("Load Chats")
    status = gr.Markdown("")
    chatbot = gr.Chatbot(type="messages")  # ✅ new format
    txt = gr.Textbox(placeholder="Say something...")
    send = gr.Button("Send")
    purge_btn = gr.Button("Purge")

    load_btn.click(load_chats, inputs=[files, state], outputs=[status, state])
    send.click(respond, inputs=[txt, chatbot, state], outputs=[chatbot, state])
    txt.submit(respond, inputs=[txt, chatbot, state], outputs=[chatbot, state])
    purge_btn.click(purge, inputs=[state], outputs=[chatbot, status, state])

if __name__ == "__main__":
    demo.launch()
