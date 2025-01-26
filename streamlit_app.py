import os
from io import BytesIO

import streamlit as st
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer


# Show title and description.
st.title("💬 Chatbot")
st.write(
    "Explore documents exploration"
)

REPLICATE_API_TOKEN = st.text_input("Secret Key", type="password")

try:
    os.mkdir('./content')
except OSError as error:
    pass

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    with open(uploaded_file.name, mode='wb') as f:
        buf = BytesIO(uploaded_file.getvalue())
        f.write(buf.getvalue())

if not REPLICATE_API_TOKEN:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
    # set the LLM
    llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
    Settings.llm = Replicate(
        model=llama2_7b_chat,
        temperature=0.01,
        additional_kwargs={"top_p": 1, "max_new_tokens": 300},
    )

    # set tokenizer to match LLM
    Settings.tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf"
    )

    # set the embed model!pip install llama-index
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    documents = SimpleDirectoryReader("./content").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
    )

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        query_engine = index.as_query_engine()
        res = query_engine.query(st.session_state.messages[-1]["content"])


        with st.chat_message("assistant"):
            response = st.write(res)
        st.session_state.messages.append({"role": "assistant", "content": response})
