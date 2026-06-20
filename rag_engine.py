from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

# Models are loaded once and reused across requests instead of being rebuilt on every question (the old search_documents() rebuilt the entire vector store on every single call).
_qa_pipeline = None
_embedding_model = None


def get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        _qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=-1,  # CPU - matches Streamlit Cloud's free tier
        )
    return _qa_pipeline


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


def load_and_chunk_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Load a PDF from disk and split it into overlapping text chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks):
    """Embed chunks and build a FAISS index for similarity search."""
    embedding_model = get_embedding_model()
    return FAISS.from_documents(chunks, embedding_model)


def answer_question(vector_store, question, k=6, score_threshold=0.0):
    """
    Retrieve the most relevant chunks for `question`, then run extractive
    QA over just those chunks (not the whole document).
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    relevant_chunks = retriever.invoke(question)

    if not relevant_chunks:
        return "Sorry, I couldn't find anything relevant to that question in the document."

    # roberta-base-squad2 has a ~512 token context limit, so cap the
    # combined context defensively rather than truncating mid-answer.
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)[:4000]

    qa = get_qa_pipeline()
    result = qa(question=question, context=context)

    if not result.get("answer") or result.get("score", 0) < score_threshold:
        return "Sorry, I couldn't find a confident answer in the document."

    return result["answer"]