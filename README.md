Document-Centric Chatbot with Local RAG
AI-Powered Regulatory Compliance Assistant

1. Introduction
   A secure, locally-run chatbot that enables natural language interaction with regulatory documents (PDFs/TXT) using:

Mistral-7B via Ollama for response generation

FAISS vector store for document retrieval

nomic-embed-text for local embeddings

MERN stack for full-stack implementation

2. Core Workflow
   Document Ingestion Phase

File type detection (PDF/TXT)

Dynamic chunking (1000 chars with 20% overlap)

Local embedding via nomic-embed-text

Query Processing Phase

Mode selection (General/Regulation)

Context-aware retrieval from FAISS

Augmented generation with Mistral-7B

Response Generation

Source citation for regulatory mode

Plain-text answers for general mode

3. Technical Implementation
   Backend Services
   Component Technology Purpose
   Document Router Express.js File upload handling
   Embedding Service LangChain + nomic Text vectorization
   Vector Store FAISS Similarity search
   LLM Gateway Ollama API Local model inference
   Frontend Components
   ModeSelector.jsx - Toggles between chat modes

DocumentPicker.jsx - Handles Text_v1/Text_v2 selection

ChatWindow.jsx - Conversation interface with source attribution

4. Usage Scenarios
   Compliance Mode
   User activates "Compliance Mode"

Chooses between Text_v1.txt or Text_v2.txt

Asks compliance-related questions

Receives answers with relevant document excerpts

General Mode
Standard chatbot functionality

No document context used

5. Assessment Alignment

| Requirement          | Implementation           | Key Features                                   |
| -------------------- | ------------------------ | ---------------------------------------------- |
| 🔒 **Local LLM**     | Ollama + Mistral-7B      | • No internet required<br>• 4-bit quantization |
| 📂 **Doc Selection** | Interactive picker       | • Two-column layout<br>• Real-time filtering   |
| 🧠 **RAG Pipeline**  | FAISS + nomic-embed-text | • 384-dimension vectors<br>• Cosine similarity |
| 💬 **Context**       | Chat history manager     | • LocalStorage backed<br>• Auto-pruning        |

6. Future Enhancements

   Batch processing for large document sets

Audit trail generation

Multi-language support

Fine-tuning for regulatory terminology
