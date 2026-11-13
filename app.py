import streamlit as st
import ollama
from rag import search, ingest
from router import route, pick_model
from tools import run_agent, WORKSPACE

st.set_page_config(page_title="PrivAI", page_icon="🔒")
st.title("🔒 PrivAI - Private Local Assistant")
st.caption("Runs fully on your laptop. No internet, no cloud.")

SYSTEM_PROMPT = (
    "You are PrivAI, a private assistant running fully offline. "
    "When context is provided, answer ONLY from that context and cite the "
    "sources like [1], [2]. If the context does not contain the answer, say "
    "you could not find it in the documents. Keep answers clear and short."
)

# ---------- Sidebar ----------
available = [m.model for m in ollama.list().models if "embed" not in m.model]
AUTO = "🤖 Auto (router)"
choice = st.sidebar.selectbox("Model", [AUTO] + available)
agent_mode = st.sidebar.toggle("🛠️ Agent mode (use tools)", value=False)
defend = st.sidebar.toggle("🛡️ Defence mode (security guard)", value=True,
                           disabled=not agent_mode)
use_docs = st.sidebar.toggle("📚 Answer from my documents (RAG)", value=True,
                             disabled=agent_mode)
top_k = st.sidebar.slider("Chunks to retrieve", 1, 8, 4)

if st.sidebar.button("🔄 Re-index docs folder"):
    with st.spinner("Indexing PDFs..."):
        ingest()
    st.sidebar.success("Done!")

if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []

with st.sidebar.expander("📁 Workspace files"):
    files = sorted(p for p in WORKSPACE.iterdir() if p.is_file())
    if not files:
        st.caption("Empty. Put files in the 'workspace' folder.")
    for p in files:
        st.download_button(f"⬇️ {p.name}", p.read_bytes(), file_name=p.name,
                           key=f"dl-{p.name}")

# ---------- Helpers ----------
def show_sources(sources):
    with st.expander("📄 Sources"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}] {s['source']} - page {s['page']}**")
            st.caption(s["text"][:300] + "...")


def show_steps(steps):
    with st.expander(f"🛠️ Tool calls ({len(steps)})"):
        for i, s in enumerate(steps, 1):
            st.markdown(f"**{i}. `{s['tool']}`**")
            st.code(str(s["args"]), language="json")
            st.caption(("⛔ " if s.get("blocked") else "") + s["result"][:300])


def build_context(question):
    """Search the vector DB and return (context_text, sources_list)."""
    try:
        hits = search(question, k=top_k)
    except Exception:
        return "", []
    sources = [{"source": m["source"], "page": m["page"], "text": d}
               for d, m, _ in hits]
    context = "\n\n".join(
        f"[{i}] ({s['source']}, page {s['page']})\n{s['text']}"
        for i, s in enumerate(sources, 1)
    )
    return context, sources


# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("route"):
            st.caption(m["route"])
        st.markdown(m["content"])
        if m.get("steps"):
            show_steps(m["steps"])
        if m.get("sources"):
            show_sources(m["sources"])

# ---------- New message ----------
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages[:-1]]

    # ===== Day 5: Agent mode =====
    if agent_mode:
        model = pick_model("agent", available) if choice == AUTO else choice
        route_text = f"🛠️ Agent: `{model}`"
        with st.chat_message("assistant"):
            st.caption(route_text)
            with st.spinner("Working with tools..."):
                reply, steps = run_agent(model, prompt, history[-4:], defend=defend)
            st.markdown(reply)
            if steps:
                show_steps(steps)
        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "steps": steps,
             "route": route_text})
        st.rerun()   # refresh so new files show up in the sidebar

    # ===== Normal mode (router + RAG) =====
    else:
        if choice == AUTO:
            task, model, reason = route(prompt, use_docs, available)
            route_text = f"🧭 Router: **{task}** → `{model}` ({reason})"
        else:
            model = choice
            route_text = f"🎛️ Manual: `{model}`"

        context, sources = build_context(prompt) if use_docs else ("", [])
        user_msg = f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
        chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + \
                        [{"role": "user", "content": user_msg}]

        def stream_reply():
            for chunk in ollama.chat(model=model, messages=chat_messages, stream=True):
                yield chunk["message"]["content"]

        with st.chat_message("assistant"):
            st.caption(route_text)
            reply = st.write_stream(stream_reply())
            if sources:
                show_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "sources": sources,
             "route": route_text})