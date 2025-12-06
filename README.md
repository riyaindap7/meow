# Meow - AI-Powered Backend

## Quick Setup

### Backend Setup

```bash
cd backend && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend && npm install
```

### Backend

```bash
cd backend && venv\Scripts\activate && python -m uvicorn backend.api.main:app --reload
```

### Frontend

```bash
cd frontend && npm start
```

## API Endpoints

### Trigger Google Drive Sync
```bash
POST http://localhost:8000/api/sync/trigger
```

### Trigger MoE Scraper
```bash
POST http://localhost:8000/api/scraper/run
```

cd backend
conda activate meowvenv
cd milvus_store
docker-compose start
cd..
(meowVenv) C:\PROJECTS\meow\backend>


python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000