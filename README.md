# Meow - AI-Powered RAG Backend with LangChain

An intelligent document retrieval and question-answering system using Vector RAG (Retrieval Augmented Generation) with LangChain, Milvus, MongoDB, and OpenRouter LLM.

## 🏗️ Architecture

- **Backend**: FastAPI with LangChain RAG pipeline
- **Vector DB**: Milvus for semantic search
- **Database**: MongoDB for conversation history and metadata
- **LLM**: OpenRouter API (Alibaba Tongyi DeepResearch 30B)
- **Authentication**: Better Auth with MongoDB adapter
- **Frontend**: Next.js 16 with React 19

## 📋 Prerequisites

- Python 3.10+ (conda environment recommended)
- Node.js 18+
- Docker (for Milvus)
- MongoDB instance
- OpenRouter API key

## 🚀 Quick Setup

### 1. Backend Setup

```bash
# Navigate to backend
cd backend

# Create and activate conda environment (recommended)
conda create -n meowVenv python=3.10
conda activate meowVenv

# Install dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd frontend
npm install
```

### 3. Environment Variables

Create `.env` files in both backend and frontend directories:

**Backend `.env`:**
```env
# MongoDB
MONGODB_URI=mongodb://192.168.0.106:27017
MONGODB_DB=victor_rag

# Milvus
MILVUS_HOST=192.168.0.107
MILVUS_PORT=19530

# OpenRouter LLM
OPENROUTER_API_KEY=your_openrouter_api_key
LLM_MODEL=alibaba/tongyi-deepresearch-30b-a3b
SITE_URL=http://localhost:3000
SITE_NAME=VICTOR

# Authentication
JWT_SECRET=your_jwt_secret
```

**Frontend `.env.local`:**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
MONGODB_URI=mongodb://192.168.0.106:27017
BETTER_AUTH_SECRET=your_better_auth_secret
```

### 4. Start Milvus (Vector Database)

```bash
cd backend/milvus_store
docker-compose up -d
```

## 🎯 Running the Application

### Start Backend Server

```bash
# Make sure conda environment is activated
conda activate meowVenv
cd backend

# Start FastAPI server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend

```bash
cd frontend
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📚 Core Dependencies

### Backend
- **FastAPI** (0.109.0+) - High-performance API framework
- **LangChain** (0.1.0+) - RAG orchestration and memory management
- **LangChain Core** (0.1.23+) - Core LangChain functionality
- **PyMilvus** (2.4.0+) - Vector database client
- **PyMongo** (4.6.0+) - MongoDB driver
- **sentence-transformers** (2.3.0+) - Text embeddings
- **httpx** (0.26.0+) - Async HTTP client for LLM API calls
- **PyJWT** (2.8.0+) - Authentication tokens
- **google-api-python-client** - Google Drive integration
- **beautifulsoup4** - Web scraping

### Frontend
- **Next.js** (16.0.3) - React framework
- **React** (19.2.0) - UI library
- **Better Auth** (1.4.5) - Authentication
- **MongoDB** (7.0.0) - Database client
- **@supabase/supabase-js** (2.81.1) - Supabase integration
- **TailwindCSS** (4.x) - Styling

## 🔧 API Endpoints

### RAG & Conversations
```bash
# Ask a question
POST /api/ask
{
  "query": "What is...",
  "conversation_id": "uuid",
  "user_id": "user123"
}

# Create conversation
POST /api/conversations
{
  "title": "New Chat",
  "user_id": "user123"
}

# Get user conversations
GET /api/conversations/{user_id}

# Get conversation messages
GET /api/conversations/{conversation_id}/messages
```

### Document Management
```bash
# Trigger Google Drive sync
POST /api/sync/trigger

# Trigger MoE scraper
POST /api/scraper/run
```

## 🧠 LangChain Features

- ✅ **Automatic Memory Management** - BaseChatMemory integration with MongoDB
- ✅ **Conversation Context** - Multi-turn conversations with history awareness
- ✅ **Vector RAG Pipeline** - Semantic search with Milvus + LLM generation
- ✅ **Source Citation** - Document references with page numbers
- ✅ **OpenRouter Integration** - Flexible LLM backend (currently Alibaba Tongyi)

## 📖 Project Structure

```
meow/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── milvus_client.py     # Vector search client
│   │   └── routers/             # API route handlers
│   ├── services/
│   │   ├── full_langchain_service.py  # TRUE LangChain RAG with memory
│   │   ├── conversation_service.py    # Conversation CRUD
│   │   ├── mongodb_service.py         # MongoDB operations
│   │   └── auth_service.py            # Authentication
│   ├── core/
│   │   ├── config.py            # App configuration
│   │   └── security.py          # Security utilities
│   ├── scripts/                 # Utility scripts
│   └── requirements.txt
├── frontend/
│   ├── app/                     # Next.js app router
│   ├── components/              # React components
│   ├── lib/                     # Utilities and clients
│   └── package.json
└── README.md
```

## 🐛 Troubleshooting

### LangChain Import Errors
If you see `ModuleNotFoundError: No module named 'langchain_core.memory'`:
```bash
# Ensure conda environment is active
conda activate meowVenv

# Verify LangChain installation
python -c "import langchain; print(langchain.__version__)"

# Should show: 0.1.0
```

### Server Not Using Conda Environment
Make sure to activate conda environment before starting server:
```bash
conda activate meowVenv  # You should see (meowVenv) in prompt
python run_server.py
```

### Milvus Connection Issues
```bash
# Check Milvus is running
docker ps | grep milvus

# Restart Milvus if needed
cd backend/milvus_store
docker-compose restart
```

## 📝 Development Notes

- Backend uses Python 3.10 in conda environment (meowVenv)
- LangChain 0.1.0 has different import paths than newer versions
- Correct imports: `from langchain.memory.chat_memory import BaseChatMemory`
- MongoDB conversations collection stores chat history
- Milvus stores document embeddings for semantic search

## 🔐 Security

- JWT-based authentication
- CORS configured for localhost development
- API key protection for LLM endpoints
- MongoDB connection with authentication support

## 📄 License

Private project - Semester7Pro