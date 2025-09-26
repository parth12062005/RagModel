# RAG System - Modal Deployment Ready

This is a modular RAG (Retrieval-Augmented Generation) system structured for Modal deployment.

## Project Structure

```
Rag_Code/
├── main.py                    # Main Modal application (deploy this to Modal)
├── src/                       # Source package
│   ├── __init__.py
│   ├── config.py             # Configuration constants and models
│   ├── models.py             # Request/Response Pydantic models
│   ├── document_processing.py # Document extraction and processing
│   ├── rag_engine.py         # RAG processing logic
│   └── helpers/              # Helper utilities
│       ├── __init__.py
│       └── utils.py          # Utility classes (logging, timing, email, API client)
├── requirements.txt          # Python dependencies
├── example_usage.py          # Example usage script
└── README.md                # This file
```

## Modal Deployment

### Deploy to Modal:

```bash
# Install Modal CLI
pip install modal

# Deploy the application
modal deploy main.py
```

### Local Development:

```bash
# Install dependencies
pip install -r requirements.txt

# Run example
python example_usage.py
```

## API Usage

Once deployed to Modal, you'll get a URL like `https://your-username--hackrx-rag-optimized-main.modal.run`

### Two-Mode Workflow

#### 1. Upload Document (Mode 1)
First, upload and process your document:

```python
import requests

# Upload document
upload_payload = {
    "document_url": "https://example.com/document.pdf",
    "config": {
        "chunking": {"chunk_size": 1024, "overlap": 250},
        "retrieval": {"semantic_search_top_k": 25, "bm25_search_top_k": 25},
        "generation": {"max_tokens": 400, "temperature": 0.1}
    },
    "email": {
        "enabled": True,
        "to_email": "your-email@example.com"  # Optional: send logs to this email
    }
}

upload_response = requests.post(
    "https://your-username--hackrx-rag-optimized-main.modal.run/hackrx/upload",
    json=upload_payload,
    headers={"Authorization": "Bearer YOUR_SECRET_TOKEN"}
)

session_data = upload_response.json()
session_id = session_data["session_id"]
print(f"Document processed! Session ID: {session_id}")
print(f"Chunks created: {session_data['chunks_count']}")
```

#### 2. Chat with Document (Mode 2)
Then ask questions using the session ID:

```python
# Ask questions
chat_payload = {
    "session_id": session_id,
    "questions": ["What is the main topic?", "What are the key findings?"],
    "config": {
        "chunking": {"chunk_size": 1024, "overlap": 250},
        "retrieval": {"semantic_search_top_k": 25, "bm25_search_top_k": 25},
        "generation": {"max_tokens": 400, "temperature": 0.1}
    },
    "email": {
        "enabled": True,
        "to_email": "your-email@example.com"  # Optional: send logs to this email
    }
}

chat_response = requests.post(
    "https://your-username--hackrx-rag-optimized-main.modal.run/hackrx/chat",
    json=chat_payload,
    headers={"Authorization": "Bearer YOUR_SECRET_TOKEN"}
)

answers = chat_response.json()["answers"]
print("Answers:", answers)
```

### Legacy Single-Endpoint Mode
You can still use the original single endpoint for backward compatibility:

```python
legacy_payload = {
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the main topic?", "What are the key findings?"],
    "config": {
        "chunking": {"chunk_size": 1024, "overlap": 250},
        "retrieval": {"semantic_search_top_k": 25, "bm25_search_top_k": 25},
        "generation": {"max_tokens": 400, "temperature": 0.1}
    },
    "email": {
        "enabled": True,
        "to_email": "your-email@example.com"  # Optional: send logs to this email
    }
}

response = requests.post(
    "https://your-username--hackrx-rag-optimized-main.modal.run/hackrx/run",
    json=legacy_payload,
    headers={"Authorization": "Bearer YOUR_SECRET_TOKEN"}
)

answers = response.json()["answers"]
```

## Key Features

- **Multi-format Document Support**: PDF, DOCX, PowerPoint, Excel, images, text, CSV
- **Hybrid Search**: Combines semantic search with BM25 keyword search
- **Configurable Parameters**: All components are configurable via Pydantic models
- **Email Logging**: Comprehensive logs sent via email
- **Performance Optimization**: Parallel processing and resource management
- **Modal Deployment**: Ready for cloud deployment with GPU support

## Configuration

All configuration is managed through `src/config.py`:

- **ChunkingConfig**: Text chunking parameters
- **RetrievalConfig**: Search and retrieval parameters
- **RRFConfig**: Reciprocal Rank Fusion weights
- **GenerationConfig**: Answer generation parameters
- **PerformanceConfig**: Performance and concurrency settings

## Endpoints

### New Two-Mode Endpoints
- `POST /hackrx/upload` - Upload and process document (returns session ID)
- `POST /hackrx/chat` - Ask questions on uploaded document (uses session ID)

### Legacy Endpoints
- `POST /hackrx/run` - Legacy single-endpoint RAG processing

### Utility Endpoints
- `GET /hackrx/config/default` - Get default configuration
- `GET /hackrx/health` - Health check

## Email Configuration

You can now specify email settings in your requests:

```python
"email": {
    "enabled": True,  # Set to False to disable email logs
    "to_email": "your-email@example.com"  # Optional: specify recipient
}
```

- If `enabled` is `True` and `to_email` is provided, logs will be sent to that email
- If `enabled` is `True` but `to_email` is not provided, logs will be sent to the default email (if configured)
- If `enabled` is `False` or not provided, no emails will be sent
