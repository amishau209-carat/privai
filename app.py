import streamlit as st
import ollama

st.set_page_config(page_title="PrivAI", page_icon="🔒")
st.title("🔒 PrivAI - Private Local Assistant")
st.caption("Runs fully on your laptop. No internet, no cloud.")

# Sidebar: pick any model you have pulled with Ollama
models = [m.model for m in ollama.list().models]
model = st.sidebar.selectbox("Model", models)
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show old messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# New message
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    def stream_reply():
        for chunk in ollama.chat(model=model,
                                 messages=st.session_state.messages,
                                 stream=True):
            yield chunk["message"]["content"]

    with st.chat_message("assistant"):
        reply = st.write_stream(stream_reply())
    st.session_state.messages.append({"role": "assistant", "content": reply})