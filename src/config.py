"""
Configuration settings for the RAG system
"""
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
load_dotenv()
# --- Model Configuration ---
EMBEDDING_MODEL_NAME = 'BAAI/bge-base-en-v1.5'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- Security Configuration ---
SECRET_TOKEN = os.getenv('SECRET_TOKEN')

# --- Groq API Configuration ---
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"

# --- Email Configuration ---
EMAIL_ENABLED = True  # Set to True to enable email sending
SMTP_SERVER = "smtp.gmail.com"  # Gmail SMTP server
SMTP_PORT = 587  # Gmail SMTP port (TLS)
EMAIL_USERNAME = "parth2sachdeva@gmail.com"  # Your email username (set via environment or config)
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')  
EMAIL_TO = "2023ucs0105@iitjammu.ac.in"  # Recipient email address (set via environment or config)

# --- File Processing Configuration ---
LARGE_PDF_THRESHOLD = 5 * 1024 * 1024  # 5MB in bytes
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB general limit
UNKNOWN_FILE_TIMEOUT = 5  # 5 seconds timeout for unknown files
UNKNOWN_FILE_MAX_SIZE = 50 * 1024 * 1024  # 50MB limit for unknown files

# --- Configuration Models ---
class ChunkingConfig(BaseModel):
    """Configuration for text chunking"""
    chunk_size: int = Field(default=1024, ge=256, le=2048)
    overlap: int = Field(default=250, ge=0, le=512)

class RetrievalConfig(BaseModel):
    """Configuration for retrieval parameters"""
    semantic_search_top_k: int = Field(default=25, ge=5, le=100)
    bm25_search_top_k: int = Field(default=25, ge=5, le=100)
    hybrid_fusion_top_k: int = Field(default=15, ge=5, le=50)
    final_context_top_k: int = Field(default=7, ge=1, le=50)

class RRFConfig(BaseModel):
    """Configuration for Reciprocal Rank Fusion"""
    semantic_weight: float = Field(default=1.0, ge=0.1, le=5.0)
    bm25_weight: float = Field(default=2.0, ge=0.1, le=5.0)
    k_parameter: int = Field(default=50, ge=1, le=100)

class GenerationConfig(BaseModel):
    """Configuration for answer generation"""
    max_tokens: int = Field(default=400, ge=50, le=1000)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.3, ge=0.1, le=1.0)

class PerformanceConfig(BaseModel):
    """Configuration for performance parameters"""
    max_search_workers: int = Field(default=16, ge=1, le=32)
    max_generation_workers: int = Field(default=10, ge=1, le=16)
    embedding_batch_size: int = Field(default=16, ge=4, le=128)
    reranking_batch_size: int = Field(default=32, ge=8, le=256)

class RAGConfig(BaseModel):
    """Complete RAG system configuration"""
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rrf: RRFConfig = Field(default_factory=RRFConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
