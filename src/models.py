"""
Pydantic models for the RAG system
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.config import RAGConfig

class EmailConfig(BaseModel):
    """Email configuration for sending logs"""
    enabled: bool = Field(default=False, description="Whether to send email logs")
    to_email: Optional[str] = Field(default=None, description="Recipient email address")

class HackRxRequest(BaseModel):
    """Request model for the RAG system (legacy - use UploadRequest or ChatRequest)"""
    documents: str = Field(..., description="URL or path to the document to process")
    questions: List[str] = Field(..., description="List of questions to answer")
    config: RAGConfig = Field(default_factory=RAGConfig, description="RAG configuration")
    email: Optional[EmailConfig] = Field(default=None, description="Email configuration")

class UploadRequest(BaseModel):
    """Request model for file upload and chunking"""
    document_url: str = Field(..., description="URL or path to the document to process")
    config: RAGConfig = Field(default_factory=RAGConfig, description="RAG configuration")
    email: Optional[EmailConfig] = Field(default=None, description="Email configuration")

class ChatRequest(BaseModel):
    """Request model for asking questions on uploaded documents"""
    session_id: str = Field(..., description="Session ID from upload response")
    questions: List[str] = Field(..., description="List of questions to answer")
    config: RAGConfig = Field(default_factory=RAGConfig, description="RAG configuration")
    email: Optional[EmailConfig] = Field(default=None, description="Email configuration")

class UploadResponse(BaseModel):
    """Response model for file upload"""
    session_id: str = Field(..., description="Session ID for future chat requests")
    chunks_count: int = Field(..., description="Number of chunks created")
    document_type: str = Field(..., description="Type of document processed")
    processing_time: float = Field(..., description="Time taken to process document")
    message: str = Field(..., description="Status message")

class ChatResponse(BaseModel):
    """Response model for chat questions"""
    answers: List[str] = Field(..., description="List of answers to the questions")
    session_id: str = Field(..., description="Session ID used")
    processing_time: float = Field(..., description="Time taken to process questions")

class HackRxResponse(BaseModel):
    """Response model for the RAG system (legacy)"""
    answers: List[str] = Field(..., description="List of answers to the questions")
