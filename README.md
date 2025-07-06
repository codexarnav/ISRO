MOSDAC/ISRO RAG Pipeline API
(Retrieval-Augmented Generation + Knowledge Graph + Whisper voice support)

📖 Overview
This project is a FastAPI-based web service that provides an AI-powered assistant for MOSDAC/ISRO data.
It combines retrieval-augmented generation (RAG), knowledge graph (KG) extraction, and voice query processing (via Whisper).
The API exposes endpoints for text and voice queries, returning rich answers and visual KG graphs.

✨ Features
🔍 RAG pipeline over MOSDAC/ISRO documents.

🗣️ Supports text and voice queries (recorded live or uploaded).

📊 Extracts entities from answers and generates knowledge graph images.

🌐 RESTful HTTP API with JSON and image responses.

🔄 Cross-Origin Resource Sharing (CORS) enabled.

🔍 Health-check endpoint.

📡 Endpoints
Method	Endpoint	Description
GET	/	Root: API version, status, endpoints list
GET	/health	Health check: verify models and data
POST	/rag	Main RAG pipeline: text or voice
POST	/rag/voice-upload	Upload audio file (wav, mp3) and process
GET	/kg-image/{filename}	Download KG image
GET	/kg-image/{filename}/base64	Get KG image as base64

🧪 Example Usage
Run a text query
bash
Copy
Edit
POST /rag
mode=text
query=What is the function of INSAT-3D?
Run a voice query (record live)
bash
Copy
Edit
POST /rag
mode=voice
duration=5
Upload an audio file
bash
Copy
Edit
POST /rag/voice-upload
audio=@query.wav
Get KG image
arduino
Copy
Edit
GET /kg-image/knowledge_graph_INSAT_3D.png
Or as base64:

bash
Copy
Edit
GET /kg-image/knowledge_graph_INSAT_3D.png/base64
🏗️ Technology Stack
FastAPI – web server framework

LangChain – RAG pipeline

Google Gemini – LLM for answering

HuggingFace Sentence Transformers – embeddings

ChromaDB – vector store

OpenAI Whisper – speech-to-text

spaCy + NLTK – entity extraction

NetworkX + Matplotlib – KG visualization

🛠️ Setup
Prerequisites
Python 3.9+

ffmpeg installed and added to PATH (for Whisper audio)

Install dependencies
bash
Copy
Edit
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Run the server
bash
Copy
Edit
uvicorn app:app --reload --host 0.0.0.0 --port 8000
The server will start on:
👉 http://localhost:8000

