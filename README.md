📄 AI Document Summarizer & RAG Q&A Bot

An intelligent AI-powered web application that allows users to upload PDF documents, generate smart summaries, and ask context-aware questions using a Retrieval-Augmented Generation (RAG) system.

Built using Streamlit, Groq LLM, FAISS, and Sentence Transformers.

🚀 Features
📄 Upload PDF documents
🧠 AI-powered automatic summarization
🔍 Ask questions from your document (RAG-based Q&A)
⚡ Fast semantic search using vector embeddings
💬 Context-aware answers using LLM (Groq)
🎯 Clean and simple Streamlit UI
🏗️ Tech Stack
Frontend: Streamlit
LLM API: Groq (LLaMA 3)
Vector Database: FAISS
Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
PDF Processing: PyPDF2
Backend Logic: Python
🧠 How It Works (RAG Pipeline)
📄 PDF is uploaded
✂️ Text is split into chunks
🔢 Each chunk is converted into embeddings
📊 FAISS stores embeddings for fast search
❓ User question is converted into embedding
🔍 Most relevant chunks are retrieved
🤖 Groq LLM generates final answer using context
📂 Project Structure

Document-QA-Bot/
│
├── app.py                # Main Streamlit application
├── requirements.txt     # Dependencies
├── README.md            # Project documentation

⚙️ Installation & Setup
1. Clone repository
git clone https://github.com/Fasiha-Iqbal/Document-QA-Bot.git
cd Document-QA-Bot
2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install dependencies
pip install -r requirements.txt
4. Add API Key

Replace in code:

client = Groq(api_key="YOUR_GROQ_API_KEY")

OR (recommended):

Create .env file:

GROQ_API_KEY=your_api_key_here
5. Run application
streamlit run app.py
📸 Demo

Upload a PDF → Get Summary → Ask Questions → Get AI Answers

🔐 Security Note
Never expose API keys in public repositories
Use .env file for sensitive credentials
Add .env to .gitignore
📈 Future Improvements
Chat history memory
Multi-document upload
Chat UI like ChatGPT
File download summaries
Cloud deployment (Streamlit Cloud / HuggingFace Spaces)
👩‍💻 Author

Fasiha Iqbal
BSc Computer Science Student
AI & Automation Enthusiast 🤖

⭐ If you like this project

Give a ⭐ on the repo — it helps a lot!