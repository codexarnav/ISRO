MOSDAC/ISRO RAG Pipeline API
An AI-powered FastAPI service that provides retrieval-augmented generation (RAG) on MOSDAC/ISRO documents, with knowledge graph extraction and voice query support.

✨ Features
✅ RAG pipeline using MOSDAC/ISRO domain documents
✅ Supports both text query and voice query (via Whisper)
✅ Extracts named entities from answers
✅ Generates and serves knowledge graphs (KG) as images
✅ Health check and CORS-enabled
✅ JSON and base64 APIs for KG image retrieval

🧩 Endpoints
Method	Endpoint	Description
GET	/	API root: version, endpoints, status
GET	/health	Health check: verify model and data loading
POST	/rag	Run full pipeline (mode=text or mode=voice)
POST	/rag/voice-upload	Upload audio file to process voice query
GET	/kg-image/{filename}	Download KG image file
GET	/kg-image/{filename}/base64	Get KG image as base64 string

🧪 Example Usage
1️⃣ Text query
bash
Copy
Edit
curl -X POST http://localhost:8000/rag \
  -F 'mode=text' \
  -F 'query=What is the function of INSAT-3D?'
2️⃣ Voice query (recorded on server)
bash
Copy
Edit
curl -X POST http://localhost:8000/rag \
  -F 'mode=voice' \
  -F 'duration=5'
3️⃣ Upload audio file
bash
Copy
Edit
curl -X POST http://localhost:8000/rag/voice-upload \
  -F 'audio=@query.wav'
4️⃣ Fetch KG image
bash
Copy
Edit
curl -O http://localhost:8000/kg-image/knowledge_graph_INSAT_3D.png
Or as base64:

bash
Copy
Edit
curl http://localhost:8000/kg-image/knowledge_graph_INSAT_3D.png/base64
🏗️ Tech Stack
FastAPI

LangChain: RAG pipeline

HuggingFace Sentence Transformers: embeddings

Google Gemini: LLM

Chroma: vector store

OpenAI Whisper: speech-to-text

spaCy + NLTK: entity extraction

NetworkX + Matplotlib: KG visualization

🛠️ Setup
🔗 Prerequisites
Python 3.9+

ffmpeg (for Whisper audio)

📦 Install dependencies
bash
Copy
Edit
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
🚀 Run server
bash
Copy
Edit
uvicorn app:app --reload --host 0.0.0.0 --port 8000
🧹 Notes
Knowledge graph images are stored in output/knowledge_graphs/

Supports CORS (*) — configure in app.py for production

Whisper model runs server-side

👨‍💻 Development
Code is organized as:

bash
Copy
Edit
/my_project/rag.py  # RAG pipeline components
app.py              # FastAPI server
Modify my_project/rag.py for prompt, models, and pipeline tweaks.

📜 License
MIT License © 2025 [codexarnav]


