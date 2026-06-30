import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
}

/* Titles */
h1, h2, h3 {
    color: #58a6ff;
}

/* Buttons */
.stButton>button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    border: none;
}

.stButton>button:hover {
    background-color: #2ea043;
}

/* Text input */
.stTextInput>div>div>input {
    background-color: #161b22;
    color: white;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background-color: #161b22;
    padding: 10px;
    border-radius: 10px;
}

/* Box styling */
.css-1d391kg {
    background-color: #161b22;
}

</style>
""", unsafe_allow_html=True)
# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Optimus AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📄 Optimus AI Document Assistant")
st.caption("Upload PDFs, get AI summaries, and ask smart questions using RAG-powered search")
st.divider()
st.markdown("""
### ⚡ AI Capabilities
✔ Smart Document Summarization  
✔ Context-aware Q&A (RAG)  
✔ Vector Search with FAISS  
✔ Groq LLM Integration  
""")
# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🤖 Optimus AI")
    st.markdown("### Smart Document Assistant")

    st.success("✔ PDF Analysis")
    st.success("✔ AI Summary")
    st.success("✔ RAG Q&A")

    st.divider()
    st.caption("Built using Streamlit + Groq + FAISS")
st.info("💡 Tip: Ask specific questions like 'summarize section 2' or 'what is the conclusion?'")
# ---------------- API ----------------
import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_ai(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ---------------- EMBEDDINGS ----------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()
# ---------------- TEXT EXTRACTION ----------------
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# ---------------- SIMPLE CHUNKING ----------------
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

# ---------------- VECTOR STORE ----------------

def create_vector_store(text):
    chunks = split_text(text)

    embeddings = embedding_model.encode(chunks, show_progress_bar=False)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    return chunks, index

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload your document (PDF only)",
    type=["pdf"],
    help="Upload any PDF and ask questions from it"
)

if uploaded_file:

    text = extract_text(uploaded_file)

    st.success("PDF uploaded successfully!")

    # RAG setup
    chunks, index = create_vector_store(text)

    st.session_state["chunks"] = chunks
    st.session_state["index"] = index

    st.subheader("📄 Document Preview")
    st.write(text[:1000])

    # SUMMARY
    with st.spinner("🤖 Generating intelligent summary..."):
        summary = ask_ai(f"""
                         Summarize the document in clear bullet points:
                         {text[:3000]}
    """)
        st.success("Summary Generated")
        st.markdown(summary)

# ---------------- Q&A (RAG) ----------------

st.subheader("🔍 Ask Anything from Document")
st.caption("AI will search relevant parts using vector search (RAG)")

question = st.text_input("Type your question")

if question:

    chunks = st.session_state.get("chunks")
    index = st.session_state.get("index")

    if not chunks or index is None:
        st.error("Please upload a PDF first.")
    else:
        with st.spinner("🧠 AI is thinking deeply..."):

            query_embedding = embedding_model.encode([question])

            distances, indices = index.search(np.array(query_embedding), 3)

            relevant_chunks = [chunks[i] for i in indices[0]]

            context = "\n".join(relevant_chunks)

            answer = ask_ai(f"""
            Answer ONLY using the context below:

            Context:
            {context}

            Question:
            {question}
            """)

        st.subheader("🤖 Answer")
        st.success("🤖 AI Response")
        st.markdown(f"""
                    > {answer}
""")
        import torch
        torch.set_num_threads(2)
        st.markdown("---")
st.markdown("🚀 Built for Optimus Automate Internship | AI Automation Project")