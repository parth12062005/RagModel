"""
Main Modal application for the RAG system
"""

import modal
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
import uuid
from typing import Dict, Any

# Local imports
from src.config import RAGConfig
from src.models import HackRxRequest, HackRxResponse
from src.rag_engine import RAGEngine

from src.config import (
    EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, SECRET_TOKEN,
    EMAIL_ENABLED, SMTP_SERVER, SMTP_PORT, EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_TO,
    RAGConfig
)
from src.models import (
    HackRxRequest, HackRxResponse, UploadRequest, UploadResponse, 
    ChatRequest, ChatResponse, EmailConfig
)
from src.helpers.utils import QuickTimer, RequestLogger, EmailSender, ResourceManager, FastGroqClient
from src.document_processing import (
    detect_file_type, safe_download_with_limits, extract_document_content,
    smart_chunk_configurable
)
from src.rag_engine import RAGEngine

# --- Modal App Setup ---
app = modal.App("hackrx-rag-optimized-new")

image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "sentence-transformers", "pymupdf", "faiss-cpu", 
        "requests", "fastapi", "uvicorn", "rank_bm25",
        "numpy", "httpx", "accelerate", "transformers", "torch",
        "email-validator", "aiosmtplib", "langchain", "langchain-text-splitters",
        "pdfplumber", "pandas", "python-docx", "python-pptx", "pillow",
        "pytesseract", "openpyxl", "xlrd", "easyocr", "opencv-python-headless","dotenv"
    ])
    .apt_install([
        "tesseract-ocr", "tesseract-ocr-eng", "libgl1-mesa-glx", "libglib2.0-0"
    ])
    .add_local_dir(
        ".",  
        remote_path="/root",
        copy=True  
    ))

@app.cls(
    image=image,
    gpu="T4",
    memory=16384,
    timeout=1200,
    keep_warm=0,
    enable_memory_snapshot=True
)
class EnhancedModelContainer:
    @modal.enter()
    def setup(self):
        """Model loading"""
        from sentence_transformers import SentenceTransformer, CrossEncoder
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.groq_client = FastGroqClient(SECRET_TOKEN)  # Using SECRET_TOKEN as API key placeholder
        self.resource_manager = ResourceManager()

        # Load models with optimizations
        print("🚀 Loading models...")
        
        # Main models
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, 
            device=self.device
        )
        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME, 
            device=self.device,
            automodel_args={"torch_dtype": torch.float16}
        )
        
        # Initialize RAG engine
        self.rag_engine = RAGEngine(
            self.embedding_model, 
            self.reranker, 
            self.groq_client, 
            self.resource_manager
        )
        
        print(f"✅ All models loaded on {self.device}!")
        
        # Initialize session storage for uploaded documents
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_email_sender(self, email_config: EmailConfig = None) -> EmailSender:
        """Get email sender based on configuration"""
        if email_config and email_config.enabled and email_config.to_email:
            return EmailSender(
                SMTP_SERVER, SMTP_PORT, EMAIL_USERNAME, EMAIL_PASSWORD, email_config.to_email
            )
        elif EMAIL_ENABLED and EMAIL_USERNAME and EMAIL_PASSWORD and EMAIL_TO:
            return EmailSender(
                SMTP_SERVER, SMTP_PORT, EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_TO
            )
        return None

    def create_session(self, chunks: list, document_type: str, config: RAGConfig) -> str:
        """Create a new session with uploaded document chunks"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "chunks": chunks,
            "document_type": document_type,
            "config": config,
            "created_at": datetime.now().isoformat()
        }
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session data by ID"""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return self.sessions[session_id]

    def extract_document_content(self, url: str, timer: QuickTimer, req_logger: RequestLogger):
        """
        Extract content from various document types based on file type detection
        """
        with timer.time_step("Document Content Extraction"):
            with timer.time_step("Download Document"):
                print(f"🔍 Downloading document from URL: {url}")
                
                # Download document
                content_bytes, content_type = safe_download_with_limits(url)
                file_size = len(content_bytes)
                
                # Detect file type
                detected_type = detect_file_type(url, content_type, content_bytes)
                req_logger.log_file_detection(url, detected_type, content_type, file_size)
                
                print(f"📄 Detected file type: {detected_type}")
                
                # Extract content based on file type
                content_blocks = extract_document_content(url, detected_type, content_bytes)
                
                # Log processing results
                req_logger.log_document_processing(
                    len(content_blocks), 
                    len(content_blocks), 
                    timer.get_total_time(), 
                    detected_type
                )
                
                return content_blocks

    def smart_chunk_documents(self, content_blocks, chunking_config, timer: QuickTimer):
        """
        Smart chunking that preserves table structure and creates overlapping chunks
        """
        with timer.time_step("Smart Chunking"):
            all_chunks = []
            
            for block in content_blocks:
                if len(block) > chunking_config.chunk_size:
                    # Split large blocks
                    block_chunks = smart_chunk_configurable(block, chunking_config)
                    all_chunks.extend(block_chunks)
                else:
                    # Keep small blocks as-is
                    all_chunks.append(block)
            
            print(f"📝 Created {len(all_chunks)} chunks from {len(content_blocks)} blocks")
            return all_chunks

    @modal.asgi_app()
    def web_app(self):
        web_app = FastAPI(title="Enhanced RAG with Email Logging", version="3.0.0")
        auth_scheme = HTTPBearer()

        def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
            if credentials.scheme != "Bearer" or credentials.credentials != SECRET_TOKEN:
                raise HTTPException(status_code=401, detail="Invalid token")

        @web_app.post("/hackrx/run", response_model=HackRxResponse)
        async def run(payload: HackRxRequest, _=Depends(verify_token)):
            import faiss
            from rank_bm25 import BM25Okapi

            timer = QuickTimer()
            req_logger = RequestLogger()
            config = payload.config  # Use provided config or defaults
            
            # Initialize resource manager with configurable workers
            self.resource_manager = ResourceManager(config.performance.max_generation_workers)
            
            # Initialize email sender if configured
            email_sender = None
            if EMAIL_ENABLED and EMAIL_USERNAME and EMAIL_PASSWORD and EMAIL_TO:
                email_sender = EmailSender(
                    SMTP_SERVER, SMTP_PORT, EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_TO
                )
                print(f"📧 Email sender configured for: {EMAIL_TO}")
            else:
                print("⚠️ Email not configured - logs will be kept in memory only")
            
            try:
                print(f"⚡ Enhanced RAG Processing: {len(payload.questions)} questions")
                
                # Log configuration and input questions
                req_logger.log_config(config)
                req_logger.log_questions_input(payload.questions)
                
                # 1. Document processing and chunking
                content_blocks = self.extract_document_content(payload.documents, timer, req_logger)
                chunks = self.smart_chunk_documents(content_blocks, config.chunking, timer)
                
                if not chunks:
                    raise ValueError("No content extracted")
                
                # 2. Build indexes with configurable parameters
                faiss_index, bm25 = self.rag_engine.build_indexes(chunks, timer, config.performance)
                
                # 3. Augmented search (query expansion + fusion)
                search_results, sanitized_questions = await self.rag_engine.augmented_parallel_search(
                    payload.questions, chunks, faiss_index, bm25, timer, req_logger,
                    config.retrieval, config.rrf, config.performance, config.generation
                )
                
                # 4. Reranking
                reranked_results = self.rag_engine.configurable_rerank(
                    payload.questions, chunks, search_results, timer, req_logger,
                    config.retrieval, config.performance
                )
                
                # 5. Build contexts from reranked results
                contexts = self.rag_engine.build_contexts(
                    payload.questions, chunks, reranked_results, timer, req_logger
                )
                
                # 6. Answer generation using sanitized questions
                answers = await self.rag_engine.configurable_generate_answers(
                    sanitized_questions, contexts, timer, req_logger,
                    config.generation, config.performance
                )
                
                # Count successful answers
                success_count = sum(1 for answer in answers if not answer.startswith("Error"))
                
                # Performance summary and final logging
                timer.print_summary()
                total_time = timer.get_total_time()
                req_logger.log_final_summary(len(payload.questions), total_time, success_count)
                
                # Send logs via email instead of saving to file
                log_content, email_sent = await req_logger.send_logs_via_email(email_sender)
                
                # Print complete logs
                print(f"\n📋 COMPLETE REQUEST LOG:")
                print(req_logger.export_logs_json())
                
                print(f"✅ Completed: {total_time:.2f}s total, {total_time/len(payload.questions):.2f}s per question")
                if email_sent:
                    print(f"📧 Detailed logs emailed to: {EMAIL_TO}")
                else:
                    print(f"💾 Logs kept in memory (size: {len(log_content)} chars)")
                
                return {"answers": answers}
                
            except Exception as e:
                req_logger.log_error("system_error", str(e))
                print(f"❌ Error: {e}")
                timer.print_summary()
                
                # Send error logs via email
                log_content, email_sent = await req_logger.send_logs_via_email(email_sender)
                
                # Print error logs
                print(f"\n📋 ERROR REQUEST LOG:")
                print(req_logger.export_logs_json())
                if email_sent:
                    print(f"📧 Error logs emailed to: {EMAIL_TO}")
                else:
                    print(f"💾 Error logs kept in memory (size: {len(log_content)} chars)")
                
                raise HTTPException(status_code=500, detail=str(e))

        @web_app.get("/hackrx/config/default")
        async def get_default_config():
            """Get the default configuration schema"""
            return RAGConfig().dict()
        
        @web_app.get("/hackrx/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "models_loaded": True
            }

        @web_app.post("/hackrx/upload", response_model=UploadResponse)
        async def upload_document(payload: UploadRequest, _=Depends(verify_token)):
            """Upload and process a document, return session ID for chat"""
            import faiss
            from rank_bm25 import BM25Okapi

            timer = QuickTimer()
            req_logger = RequestLogger()
            config = payload.config
            
            # Initialize email sender if configured
            email_sender = self.get_email_sender(payload.email)
            if email_sender:
                print(f"📧 Email sender configured for: {payload.email.to_email if payload.email else EMAIL_TO}")
            else:
                print("⚠️ Email not configured - logs will be kept in memory only")
            
            try:
                print(f"📄 Uploading and processing document: {payload.document_url}")
                
                # Log configuration
                req_logger.log_config(config)
                req_logger.log_document_input(payload.document_url)
                
                # 1. Document processing and chunking
                content_blocks = self.extract_document_content(payload.document_url, timer, req_logger)
                chunks = self.smart_chunk_documents(content_blocks, config.chunking, timer)
                
                if not chunks:
                    raise ValueError("No content extracted from document")
                
                # 2. Build indexes
                faiss_index, bm25 = self.rag_engine.build_indexes(chunks, timer, config.performance)
                
                # 3. Create session
                document_type = req_logger.logs[-1].get("detected_file_type", "UNKNOWN") if req_logger.logs else "UNKNOWN"
                session_id = self.create_session(chunks, document_type, config)
                
                # Store indexes in session for later use
                self.sessions[session_id]["faiss_index"] = faiss_index
                self.sessions[session_id]["bm25"] = bm25
                
                # Performance summary
                timer.print_summary()
                total_time = timer.get_total_time()
                req_logger.log_final_summary(0, total_time, 0)  # No questions yet
                
                # Send logs via email if configured
                log_content, email_sent = await req_logger.send_logs_via_email(email_sender)
                
                print(f"✅ Document processed: {len(chunks)} chunks in {total_time:.2f}s")
                print(f"📋 Session ID: {session_id}")
                if email_sent:
                    print(f"📧 Processing logs emailed")
                
                return UploadResponse(
                    session_id=session_id,
                    chunks_count=len(chunks),
                    document_type=document_type,
                    processing_time=total_time,
                    message=f"Document processed successfully. {len(chunks)} chunks created."
                )
                
            except Exception as e:
                req_logger.log_error("upload_error", str(e))
                print(f"❌ Upload error: {e}")
                timer.print_summary()
                
                # Send error logs via email
                log_content, email_sent = await req_logger.send_logs_via_email(email_sender)
                
                raise HTTPException(status_code=500, detail=str(e))

        @web_app.post("/hackrx/chat", response_model=ChatResponse)
        async def chat_with_document(payload: ChatRequest, _=Depends(verify_token)):
            """Ask questions on an uploaded document using session ID"""
            timer = QuickTimer()
            req_logger = RequestLogger()
            
            # Initialize email sender if configured
            email_sender = self.get_email_sender(payload.email)
            if email_sender:
                print(f"📧 Email sender configured for: {payload.email.to_email if payload.email else EMAIL_TO}")
            else:
                print("⚠️ Email not configured - logs will be kept in memory only")
            
            try:
                print(f"💬 Chat session: {payload.session_id} with {len(payload.questions)} questions")
                
                # Get session data
                session_data = self.get_session(payload.session_id)
                chunks = session_data["chunks"]
                faiss_index = session_data["faiss_index"]
                bm25 = session_data["bm25"]
                config = payload.config
                
                # Log configuration and questions
                req_logger.log_config(config)
                req_logger.log_questions_input(payload.questions)
                
                # 1. Augmented search
                search_results, sanitized_questions = await self.rag_engine.augmented_parallel_search(
                    payload.questions, chunks, faiss_index, bm25, timer, req_logger,
                    config.retrieval, config.rrf, config.performance, config.generation
                )
                
                # 2. Reranking
                reranked_results = self.rag_engine.configurable_rerank(
                    payload.questions, chunks, search_results, timer, req_logger,
                    config.retrieval, config.performance
                )
                
                # 3. Build contexts
                contexts = self.rag_engine.build_contexts(
                    payload.questions, chunks, reranked_results, timer, req_logger
                )
                
                # 4. Answer generation using sanitized questions
                answers = await self.rag_engine.configurable_generate_answers(
                    sanitized_questions, contexts, timer, req_logger,
                    config.generation, config.performance
                )
                
                # Count successful answers
                success_count = sum(1 for answer in answers if not answer.startswith("Error"))
                
                # Performance summary
                timer.print_summary()
                total_time = timer.get_total_time()
                req_logger.log_final_summary(len(payload.questions), total_time, success_count)
                
                # Send logs via email if configured
                log_content, email_sent = await req_logger.send_logs_via_email(email_sender)
                
                print(f"✅ Chat completed: {success_count}/{len(payload.questions)} questions answered in {total_time:.2f}s")
                if email_sent:
                    print(f"📧 Chat logs emailed")
                
                return ChatResponse(
                    answers=answers,
                    session_id=payload.session_id,
                    processing_time=total_time
                )
                
            except Exception as e:
                req_logger.log_error("chat_error", str(e))
                print(f"❌ Chat error: {e}")
                timer.print_summary()
                
                # Send error logs via email
                log_content, email_sent = await req_logger.send_logs_via_email(email_sender)
                
                raise HTTPException(status_code=500, detail=str(e))

        return web_app
