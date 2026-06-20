import streamlit as st
import tempfile
import os
import re
import PyPDF2

from rag_engine import load_and_chunk_pdf, build_vector_store, answer_question

st.title("Smart PDF Question Answering App")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Only rebuild the index when a NEW file is uploaded. The old version rebuilt the FAISS vector store from scratch on every single question, which is slow and unnecessary.
    if st.session_state.get("uploaded_filename") != uploaded_file.name:
        with st.spinner("Reading and indexing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            chunks = load_and_chunk_pdf(tmp_path)
            st.session_state["vector_store"] = build_vector_store(chunks)
            os.unlink(tmp_path)

            uploaded_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            st.session_state["raw_text"] = text
            st.session_state["uploaded_filename"] = uploaded_file.name

    st.subheader("Extracted Document Content:")
    st.text_area("Text from PDF", value=st.session_state["raw_text"], height=300)

    SECTION_HEADERS = [
        "SUMMARY", "EDUCATION", "SKILLS", "PROJECTS",
        "ACHIEVEMENTS", "OBJECTIVE", "EXPERIENCE",
    ]

    def handle_special_queries(question, context):
        """Fast-path common structured queries without invoking the model."""
        question_lower = question.lower()
        for header in SECTION_HEADERS:
            if header.lower() in question_lower:
                other_headers = "|".join(h for h in SECTION_HEADERS if h != header)
                pattern = rf"{header}\s*\n(.*?)(?=\n(?:{other_headers})\b|\Z)"
                match = re.search(pattern, context, re.DOTALL)
                if match and match.group(1).strip():
                    return match.group(1).strip()
                return f"Sorry, I couldn't find the {header.capitalize()} section in the document."
        return None

    st.subheader("Ask a Question About the Document:")
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question.")
        else:
            special_answer = handle_special_queries(question, st.session_state["raw_text"])
            if special_answer:
                st.success(f"Answer: {special_answer}")
            else:
                with st.spinner("Searching document and generating answer..."):
                    try:
                        answer = answer_question(st.session_state["vector_store"], question)
                        st.success(f"Answer: {answer}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")