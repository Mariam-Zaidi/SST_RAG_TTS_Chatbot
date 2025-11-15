**🎙️ Multi-User Voice-Enabled RAG Chatbot**

Speech-to-Text → Retrieval-Augmented Generation → Text-to-Speech
Supports PDFs, text files & voice ingestion • Personalized namespaces • Pinecone vector storage • Ollama LLM backend


**📌 Overview**

This project is a multi-user Retrieval-Augmented Generation (RAG) chatbot that supports:

✅ Voice input (STT) using Whisper + sounddevice
✅ Document ingestion (PDF, TXT, voice transcripts)
✅ Vector storage per-user using Pinecone namespaces
✅ Context-aware chat using LangChain Conversational Memory
✅ LLM answering using Ollama (Mistral / Llama3 / etc.)
✅ Voice output (TTS) for bot responses
✅ Supports multiple users simultaneously

**The workflow:**

User → (Speak) → Whisper STT → Query Vectorstore → Ollama LLM → TTS → Bot Speaks Response


You can ingest files or voice notes into a user’s private vector namespace, then ask questions verbally or textually.

**✨ Features**

Feature	Description

🎤 STT Input	Whisper transcribes voice recordings (sounddevice)
📄 Document ingestion	PDFs/TXT split into chunks and stored in Pinecone
🔎 RAG retrieval	LangChain + Pinecone vector search
🧠 Per-User Memory	Each user has their own conversation buffer + namespace
🗂️ Pinecone VectorDB	Stores embeddings for all users separately
🤖 LLM Responses	Powered by Ollama + LangChain
🔊 TTS	Bot responses are spoken
👥 Multi-User Support	user_id creates isolated namespaces


**🏗️ Project Architecture**


                           ┌─────────────────────┐
                           │     User (Voice)     │
                           └──────────┬───────────┘
                                      │ speak
                                      ▼
                         ┌──────────────────────────┐
                         │     Whisper STT (local)  │
                         └──────────┬───────────────┘
                                      │ text
                                      ▼
                     ┌────────────────────────────────────┐
                     │      ConversationalRetrievalChain   │
                     └──────────┬──────────────────────────┘
                                │ retrieves
                                ▼
                        ┌──────────────────┐
                        │  Pinecone Vectors│◄── Ingest PDFs / Voice
                        └──────────────────┘
                                │ context
                                ▼
                     ┌────────────────────────────────────┐
                     │       Ollama LLM (Mistral etc.)     │
                     └──────────┬──────────────────────────┘
                                │ answer
                                ▼
                         ┌──────────────────────────┐
                         │         TTS Output       │
                         └──────────────────────────┘
