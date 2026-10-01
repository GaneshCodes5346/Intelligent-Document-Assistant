import os, re, warnings, logging
warnings.simplefilter("ignore", DeprecationWarning)

from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain-community.*")
logging.getLogger("google_genai").setLevel(logging.ERROR)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=api_key, max_output_tokens=1000)

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text

def load_doc(path):
    doc = Document(path)
    text = "\n".join(para.text for para in doc.paragraphs)

    return text

def load_file(path):
    if path.endswith(".pdf"):
        return load_pdf(path)
    elif path.endswith(".docx"):
        return load_doc(path)
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {path}")

path = input("Enter the path to your document (.pdf, .docx, or .txt): ").strip('"').strip()

while not os.path.exists(path):
    print("That file doesn't exist. Try again.")
    path = input("Enter the path to your document: ").strip('"').strip()

doc_text = load_file(path)
doc_text = re.sub(r'\n{3,}', '\n\n', doc_text)
doc_text = re.sub(r'[ \t]{2,}', ' ', doc_text)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 30
)
chunks = splitter.split_text(doc_text)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key = api_key)


vector_store = FAISS.from_texts(chunks, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful document analyzer bot. Explain only in 2 sentences using the context. \n context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["input"]))) | prompt | llm | StrOutputParser()
)

store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

bot_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    result = bot_with_memory.invoke(
        {"input":user_input},
        config={"configurable":{"session_id":"user_1"}}
    )
    print("Bot: ", result)
