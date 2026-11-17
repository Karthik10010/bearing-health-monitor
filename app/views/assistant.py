import streamlit as st

from components.common import get_report
from pdm.llm import gemini_client
from pdm.llm.chat import answer
from pdm.llm.context import build_context
from pdm.llm.fallback import QUESTION_TYPES

st.title("AI Assistant")
if gemini_client.available():
    st.success("Mode: **Google Gemini** — answers use only the analysis report shown below.")
else:
    st.info("Mode: **offline templates** — no GEMINI_API_KEY in .env. Answers are filled with the real report "
            "values. Add a key to enable free-form Gemini answers.")

rep = get_report()
if rep is not None:
    st.session_state.setdefault("chat", [])
    qcols = st.columns(4)
    clicked = None
    for i, q in enumerate(QUESTION_TYPES.values()):
        if qcols[i % 4].button(q, key=f"q{i}", use_container_width=True):
            clicked = q
    typed = st.chat_input("Ask about this bearing...")
    question = typed or clicked

    for turn in st.session_state["chat"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            for note in turn.get("notes", []):
                st.caption(note)

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        history = [{"role": t["role"], "content": t["content"]} for t in st.session_state["chat"]]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                a = answer(question, rep, history)
            st.markdown(a.text)
            notes = [f"Answered by: {'Gemini' if a.mode == 'gemini' else 'offline template'}"]
            if a.note:
                notes.append(a.note)
            if a.unsupported_numbers:
                notes.append("⚠ These numbers in the answer were NOT found in the report data — treat them with "
                             "caution: " + ", ".join(a.unsupported_numbers))
            for n in notes:
                st.caption(n)
        st.session_state["chat"] += [{"role": "user", "content": question},
                                     {"role": "assistant", "content": a.text, "notes": notes}]
    if st.session_state["chat"] and st.button("Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()

    with st.expander("Exactly what data the AI receives (nothing else)"):
        st.json(build_context(rep), expanded=False)
