from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate, MessagesPlaceholder
import pyttsx3
from langchain_huggingface import HuggingFaceEmbeddings
from collections import defaultdict
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# ---------------- Load environment ----------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# ---------------- Config ----------------
WHISPER_MODEL = "base"
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
model = whisper.load_model(WHISPER_MODEL)

index_name = "multiuser-rag"

# ---------------- Pinecone client ----------------
pc = Pinecone(api_key=PINECONE_API_KEY)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# ---------------- Embeddings + LLM ----------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOllama(model="mistral")

# ---------------- Multi-user memory ----------------
user_memories = defaultdict(lambda: ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    return_messages=True,
    output_key="answer"  # store only answer
))

# ---------------- TTS Setup ----------------
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)
tts_engine.setProperty("volume", 1.0)

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# ---------------- Helper Functions ----------------
def ingest_voice(user_id):
    SAMPLE_RATE = 16000
    DURATION = 25

    print(f"🎤 Recording for {DURATION} seconds...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    audio = audio.flatten()
    wav.write("temp.wav", SAMPLE_RATE, audio)

    result = model.transcribe("temp.wav", fp16=False)
    text = result["text"].strip()
    if not text:
        print("❌ No speech detected.")
        return
    print("📝 Transcribed:", text)

    docs = [Document(page_content=text)]
    print(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
    chunks = splitter.split_documents(docs)

    #test_emb = embeddings.embed_query("hello world")
    #print(len(test_emb))  # should be 384

    vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    namespace=user_id
    )
    
    ids = [f"{user_id}-voice-{i}" for i in range(len(chunks))]
    vectorstore.add_documents(chunks, ids=ids)

    # Check Pinecone stats
    stats = index.describe_index_stats()
    print("📊 Pinecone stats:", stats)
    
    print(f"✅ Inserted {len(chunks)} voice chunks for {user_id}")


def ingest_docs(user_id, file_path):
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("❌ Only .txt and .pdf files are supported.")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(chunks)

    vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    namespace=user_id
    )
    vectorstore.add_documents(chunks)

    print(f"✅ Ingested {len(chunks)} chunks for user {user_id}")
    
def get_retriever(user_id, k=3):
    vectorstore = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name,
        namespace=user_id
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "lambda_mult": 0.5}
    )  

def get_chain(user_id):

    # simple human prompt, let LangChain inject context automatically
    qa_prompt = PromptTemplate(
        template="""You are a helpful assistant.
        Use the following context to answer the question.
        If you don't know, say you don't know.

        Context: {context}
        Question: {question}
        Answer:""",
        input_variables=["context", "question"],
        )

    retriever = get_retriever(user_id)

    # Debug: test retrieval directly
    test_docs = retriever.invoke("test query")
    if test_docs:
        print("🔎 Retrieved sample docs:")
        for d in test_docs:
            print("   ", d.page_content[:120])
    else:
        print("⚠️ No docs retrieved for test query")


    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=user_memories[user_id],
        combine_docs_chain_kwargs={
            "prompt": qa_prompt,
            "document_variable_name": "context"  # explicitly tell it
        },
        return_source_documents=True,
        output_key="answer"
    )
    return chain

# ---------------- Chat Loop ----------------
print("📌 Multi-user Chatbot ready!")
print("👉 Commands:")
print("   ingest <user_id> <file_path>")
print("   voice <user_id>")
print("   <user_id>: <message>")
print("   exit")

while True:
    query = input(">>> ").strip()
    if query.lower() == "exit":
        break

    # ---------------- Voice ingestion ----------------
    if query.startswith("voice"):
        try:
            _, user_id = query.split(" ", 1)
            ingest_voice(user_id.strip())
        except Exception as e:
            print("❌ Usage: voice <user_id>")
            print("Error:", e)
        continue

    # ---------------- Document ingestion ----------------
    if query.startswith("ingest"):
        try:
            _, user_id, file_path = query.split(" ", 2)
            ingest_docs(user_id.strip(), file_path.strip())
        except Exception as e:
            print("❌ Usage: ingest <user_id> <file_path>")
            print("Error:", e)
        continue

    # ---------------- Chat message ----------------
    try:
        user_id, question = query.split(":", 1)
        user_id, question = user_id.strip(), question.strip()
    except:
        print("❌ Format must be '<user_id>: <message>' or 'ingest <user_id> <file_path>'")
        continue

    # Retrieve + Answer
    chain = get_chain(user_id)
    result = chain.invoke({"question": question})
    print("Generating answer....")
    answer = result["answer"]

    # Print + TTS
    print(f"{user_id} -> Bot:", answer)
    speak(answer)
