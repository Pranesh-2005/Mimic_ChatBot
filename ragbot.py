import gradio as gr
import os, re, datetime, collections
import chromadb
from chromadb.utils import embedding_functions
from dataclasses import dataclass, field
from openai import AzureOpenAI   # ✅ new import

# Configure Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2025-01-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://YOUR-RESOURCE.openai.azure.com/
)

# Chroma for embeddings
chroma_client = chromadb.Client()
embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_type="azure",
    model_name="text-embedding-ada-002"
)

@dataclass
class SessionState:
    chroma_dbs: dict = field(default_factory=dict)  # persona_name -> chroma collection
    personas: list = field(default_factory=list)

def parse_whatsapp_text(text: str):
    messages = []
    for line in text.splitlines():
        if " - " in line and ":" in line:
            try:
                date_time, rest = line.split(" - ", 1)
                sender, msg = rest.split(":", 1)
                messages.append((sender.strip(), msg.strip()))
            except:
                continue
    return messages

def load_chats(files, state: SessionState):
    persona_msgs = collections.defaultdict(list)
    for f in files or []:
        try:
            txt = open(f, encoding="utf-8").read()
        except:
            txt = open(f, encoding="utf-16").read()
        for sender, msg in parse_whatsapp_text(txt):
            persona_msgs[sender].append(msg)

    state.personas = list(persona_msgs.keys())

    # Build per-person Chroma collection
    for sender, msgs in persona_msgs.items():
        collection = chroma_client.create_collection(name=f"{sender}_{id(state)}", embedding_function=embedder)
        for i, m in enumerate(msgs):
            collection.add(documents=[m], ids=[f"{sender}_{i}"])
        state.chroma_dbs[sender] = collection

    return gr.Dropdown(choices=state.personas, value=(state.personas[0] if state.personas else None)), f"Loaded {sum(len(v) for v in persona_msgs.values())} messages", state

def respond(user_input, persona_name, chat_history, state: SessionState):
    if not persona_name or persona_name not in state.chroma_dbs:
        return chat_history, state

    collection = state.chroma_dbs[persona_name]
    results = collection.query(query_texts=[user_input], n_results=5)

    context = "\n".join(results["documents"][0]) if results["documents"] else ""
    system_prompt = f"You are mimicking {persona_name} based on WhatsApp style. Reply casually like them."

    completion = client.chat.completions.create(   # ✅ new API
        model="gpt-4.1",  # or your Azure deployment name
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User said: {user_input}\n\nRelevant past messages:\n{context}\n\nReply in tone of {persona_name}:"}
        ]
    )
    reply = completion.choices[0].message.content   # ✅ new response format

    chat_history = chat_history + [(user_input, reply)]
    return chat_history, state

def purge(state: SessionState):
    state.chroma_dbs.clear()
    state.personas.clear()
    return None, "Purged session data.", state

with gr.Blocks() as demo:
    gr.Markdown("# 🦜 WhatsApp Mimic Chat (Azure OpenAI + RAG)")
    state = gr.State(SessionState())

    files = gr.File(label="Upload WhatsApp .txt", file_count="multiple", type="filepath")
    persona = gr.Dropdown(label="Choose persona")
    load_btn = gr.Button("Build personas")
    status = gr.Markdown("")
    chatbot = gr.Chatbot()
    txt = gr.Textbox(placeholder="Say something...")
    send = gr.Button("Send")
    purge_btn = gr.Button("Purge")

    load_btn.click(load_chats, inputs=[files, state], outputs=[persona, status, state])
    send.click(respond, inputs=[txt, persona, chatbot, state], outputs=[chatbot, state])
    txt.submit(respond, inputs=[txt, persona, chatbot, state], outputs=[chatbot, state])
    purge_btn.click(purge, inputs=[state], outputs=[chatbot, status, state])

if __name__ == "__main__":
    demo.launch()
