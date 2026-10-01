# Intelligent Document Assistant

A command-line RAG (Retrieval-Augmented Generation) chatbot that answers questions about any PDF, Word, or text document you give it — with conversational memory across the session.

## What it does

- Load any `.pdf`, `.docx`, or `.txt` file by entering its path
- Splits the document into chunks and embeds them with Google's Gemini embedding model
- Stores and searches those embeddings with FAISS for fast, relevant retrieval
- Answers questions using only the retrieved context — concise, 2-sentence answers
- Remembers the conversation, so follow-up questions like ("what was my first question?") work correctly

## Example

```
Enter the path to your document (.pdf, .docx, or .txt): "Ganesh_Resume.pdf"
You: what is my name in the document
Bot: Based on the provided document, your name is Ganesh Dakoju. It is listed at the very top of your profile header.
You: Correct. What is my publication then?
Bot: Your publication is titled "An Integrated YOLO-Based Machine Learning Framework for Automated Road Damage Detection and Classification." It was presented at the 2nd IEEE International Conference on Next Generation Communication & Information Processing (INCIP 2026).
You: what is my first question?
Bot: Your first question was, "what is my name in the document." You asked this at the start of our conversation.
```

## Tech stack

- **LangChain** — orchestration (prompt templates, chains, conversation memory)
- **Google Gemini API** — `gemini-3.6-flash` for generation, `gemini-embedding-001` for embeddings
- **FAISS** — in-memory vector similarity search
- **pypdf** / **python-docx** — document parsing

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
3. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
4. Run it:
   ```bash
   python main.py
   ```
5. Enter the path to any PDF, Word, or text file when prompted, then start asking questions. Type `exit` to quit.

## How it works

1. **Load** — reads the file based on its extension (PDF via `pypdf`, Word via `python-docx`, plain text via built-in file handling)
2. **Split** — breaks the document into ~500-character chunks with overlap, using LangChain's `RecursiveCharacterTextSplitter`
3. **Embed & store** — each chunk is embedded and indexed in a fresh FAISS vector store, rebuilt on every run so switching documents never mixes up results from a previous one
4. **Retrieve** — on each question, the most relevant chunks are pulled from the FAISS index
5. **Generate** — retrieved chunks + conversation history are passed to Gemini, which answers grounded only in that context

## Possible next steps

- Surface source chunks/page numbers alongside each answer
- Support multiple documents in a single session
- Swap the CLI for a simple web UI
