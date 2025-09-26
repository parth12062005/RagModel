"""
Utility functions and classes for the RAG system
"""
import logging
import time
import hashlib
import threading
import smtplib
import aiosmtplib
from datetime import datetime
from contextlib import contextmanager
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import List, Dict, Any
import httpx
import uuid
import asyncio
import json

from ..config import RAGConfig, GenerationConfig, LARGE_PDF_THRESHOLD, GROQ_API_URL, GROQ_MODEL, GROQ_API_KEY

def setup_logging():
    """Setup comprehensive logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class EmailSender:
    """Email sender for log files"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, to_email: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.to_email = to_email
    
    async def send_log_email(self, log_content: str, request_id: str) -> bool:
        """Send log content via email as attachment"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = self.to_email
            msg['Subject'] = f"RAG Pipeline Log - {request_id}"
            
            # Email body
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            body = f"""
RAG Pipeline Processing Complete

Request ID: {request_id}
Timestamp: {timestamp}

The detailed log file is attached to this email.

This log contains:
- Complete request tracking and timing
- Top 7 context chunks for each question
- Performance metrics and statistics
- All search and generation results
- Error information (if any)

Best regards,
RAG Pipeline System
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach log file
            log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            attachment = MIMEApplication(log_content.encode('utf-8'), _subtype='txt')
            attachment.add_header('Content-Disposition', 'attachment', filename=log_filename)
            msg.attach(attachment)
            
            # Send email asynchronously
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_server,
                port=self.smtp_port,
                start_tls=True,
                username=self.username,
                password=self.password,
                timeout=30
            )
            
            print(f"📧 Log email sent successfully to {self.to_email}")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to send log email: {e}")
            return False
    
    def send_log_email_sync(self, log_content: str, request_id: str) -> bool:
        """Synchronous version of email sending"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = self.to_email
            msg['Subject'] = f"RAG Pipeline Log - {request_id}"
            
            # Email body
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            body = f"""
RAG Pipeline Processing Complete

Request ID: {request_id}
Timestamp: {timestamp}

The detailed log file is attached to this email.

This log contains:
- Complete request tracking and timing
- Top 7 context chunks for each question
- Performance metrics and statistics
- All search and generation results
- Error information (if any)

Best regards,
RAG Pipeline System
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach log file
            log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            attachment = MIMEApplication(log_content.encode('utf-8'), _subtype='txt')
            attachment.add_header('Content-Disposition', 'attachment', filename=log_filename)
            msg.attach(attachment)
            
            # Send email synchronously
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            print(f"📧 Log email sent successfully to {self.to_email}")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to send log email: {e}")
            return False

class RequestLogger:
    """Enhanced logging for request tracking and debugging"""
    
    def __init__(self):
        self.logs = []
        self.start_time = datetime.now()
        self.req_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def log_config(self, config: RAGConfig):
        """Log the configuration being used"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "configuration",
            "request_id": self.req_id,
            "config": config.dict()
        }
        self.logs.append(log_entry)
        logger.info(f"[{self.req_id}] CONFIG: Using custom configuration")
    
    def log_document_input(self, document_url: str, pdf_size: int = None):
        """Log input document details"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "document_input",
            "request_id": self.req_id,
            "document_url": document_url,
            "pdf_size_mb": round(pdf_size / (1024 * 1024), 2) if pdf_size else None,
            "extraction_method": "text_only" if pdf_size and pdf_size > LARGE_PDF_THRESHOLD else "full_with_tables"
        }
        self.logs.append(log_entry)
        logger.info(f"[{self.req_id}] DOCUMENT INPUT: {document_url} ({log_entry['pdf_size_mb']}MB) - Method: {log_entry['extraction_method']}")
    
    def log_questions_input(self, questions: List[str]):
        """Log input questions"""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "event": "questions_input",
            "questions": questions,
            "question_count": len(questions)
        })

    def log_document_processing(self, chunks_extracted: int, total_blocks: int, processing_time: float, doc_type: str = "PDF", 
                               additional_info: Dict = None):
        """Log document processing results for any document type"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "document_processing",
            "request_id": self.req_id,
            "document_type": doc_type,
            "total_blocks": total_blocks,
            "chunks_extracted": chunks_extracted,
            "processing_time_seconds": round(processing_time, 2),
            "additional_info": additional_info or {}
        }
        self.logs.append(log_entry)
        logger.info(f"[{self.req_id}] {doc_type} PROCESSING: {total_blocks} blocks → {chunks_extracted} chunks in {processing_time:.2f}s")
    
    def log_file_detection(self, url: str, detected_type: str, content_type: str = None, file_size: int = None):
        """Log file type detection results"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "file_detection",
            "request_id": self.req_id,
            "detected_file_type": detected_type,
            "content_type": content_type,
            "file_size_mb": round(file_size / (1024 * 1024), 2) if file_size else None
        }
        self.logs.append(log_entry)
        logger.info(f"[{self.req_id}] FILE DETECTION: {detected_type} ({log_entry['file_size_mb']}MB)")
    
    def log_top_contexts(self, question_index: int, question: str, context_chunks: List[str]):
        """Log top context chunks for a question"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "top_contexts",
            "request_id": self.req_id,
            "question_index": question_index,
            "question": question,
            "context_count": len(context_chunks),
            "contexts": context_chunks[:7]  # Top 7 contexts
        }
        self.logs.append(log_entry)
        logger.info(f"[{self.req_id}] TOP CONTEXTS Q{question_index}: {len(context_chunks)} chunks")
    
    def log_augmented_queries(self, question_index: int, question: str, augmented_queries: List[str]):
        """Log augmented queries generated for a question"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "augmented_queries",
            "request_id": self.req_id,
            "question_index": question_index,
            "question": question,
            "augmented_query_count": len(augmented_queries or []),
            "augmented_queries": augmented_queries or []
        }
        self.logs.append(log_entry)
        logger.info(f"[{self.req_id}] AUGMENTED QUERIES Q{question_index}: {len(augmented_queries or [])} variants")
    
    def log_final_summary(self, total_questions: int, total_time: float, success_count: int):
        """Log final processing summary"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "final_summary",
            "request_id": self.req_id,
            "total_questions": total_questions,
            "total_time_seconds": round(total_time, 2),
            "success_count": success_count,
            "success_rate": round(success_count / total_questions * 100, 1) if total_questions > 0 else 0
        }
        self.logs.append(log_entry)
        logger.info(f"[{self.req_id}] FINAL SUMMARY: {success_count}/{total_questions} questions answered in {total_time:.2f}s")
    
    def log_error(self, error_type: str, error_message: str):
        """Log error information"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "request_id": self.req_id,
            "error_type": error_type,
            "error_message": error_message
        }
        self.logs.append(log_entry)
        logger.error(f"[{self.req_id}] ERROR {error_type}: {error_message}")
    
    def export_logs_json(self) -> str:
        """Export logs as JSON string"""
        return json.dumps({
            "request_id": self.req_id,
            "start_time": self.start_time.isoformat(),
            "logs": self.logs
        }, indent=2)
    
    async def send_logs_via_email(self, email_sender: EmailSender = None) -> tuple[str, bool]:
        """Send logs via email and return log content and success status"""
        log_content = self.export_logs_json()
        
        if email_sender:
            email_sent = await email_sender.send_log_email(log_content, self.req_id)
            return log_content, email_sent
        else:
            return log_content, False

    def log_search_results(self, question_idx: int, search_results: Dict[str, Any], search_type: str):
        """Log search results for a specific question"""
        # Ensure search_results is a dictionary before accessing with .get()
        if isinstance(search_results, str):
            search_results = {"error": search_results}
            
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "event": "search_results",
            "question_index": question_idx,
            "search_type": search_type,
            "top_k_indices": search_results.get("indices", []),
            "top_k_scores": search_results.get("scores", []),
            "processing_time": search_results.get("time", 0)
        })
        logger.info(f"[{self.req_id}] SEARCH RESULTS: Question {question_idx}, Type: {search_type}")

    def log_answer_generated(self, question_idx: int, question: str, answer: str, context_length: int):
        """Log generated answer and its details"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "answer_generated",
            "request_id": self.req_id,
            "question_index": question_idx,
            "question": question,
            "answer": answer,
            "context_length": context_length
        }
        self.logs.append(log_entry)
        logger.info(f"[{self.req_id}] ANSWER GENERATED: Question {question_idx}, Context Length: {context_length}")


class QuickTimer:
    """Simple timer for performance tracking"""
    def __init__(self):
        self.steps = []
        self.start_time = datetime.now()

    @contextmanager
    def time_step(self, step_name: str):
        start = datetime.now()
        try:
            yield
        finally:
            duration = (datetime.now() - start).total_seconds()
            self.steps.append((step_name, duration))

    def get_total_time(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    def print_summary(self):
        print("\n⏱️ Timing Summary:")
        for step, duration in self.steps:
            print(f"  • {step}: {duration:.2f}s")
        print(f"  Total: {self.get_total_time():.2f}s")

class ResourceManager:
    """Thread-safe resource manager for concurrent operations"""
    def __init__(self, max_workers: int = 4):
        self.embedding_semaphore = asyncio.Semaphore(1)  # Limit embedding to 1 at a time
        self.reranker_semaphore = asyncio.Semaphore(1)  # Limit reranking to 1 at a time
        self.groq_semaphore = asyncio.Semaphore(max_workers)  # Configurable LLM concurrency

    @contextmanager
    def embedding_context(self):
        try:
            self.embedding_semaphore.acquire()
            yield
        finally:
            self.embedding_semaphore.release()

    @contextmanager
    def reranker_context(self):
        try:
            self.reranker_semaphore.acquire()
            yield
        finally:
            self.reranker_semaphore.release()

    @contextmanager
    def groq_context(self):
        try:
            self.groq_semaphore.acquire()
            yield
        finally:
            self.groq_semaphore.release()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup any remaining resources
        if self.embedding_semaphore._value < 1:
            self.embedding_semaphore.release()
        if self.reranker_semaphore._value < 1:
            self.reranker_semaphore.release()
        if self.groq_semaphore._value < self.groq_semaphore._bound:
            self.groq_semaphore.release()

class FastGroqClient:
    """Client for Groq API with error handling"""
    def __init__(self, api_key: str):
        self.client = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=60.0
        )

    async def generate_answer(self, prompt: str, generation_config: Any) -> str:
        try:
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": generation_config.temperature,
                    "max_tokens": generation_config.max_tokens,
                    "top_p": generation_config.top_p
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise ValueError(f"Groq API error: {response.status_code}")
