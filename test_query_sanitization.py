#!/usr/bin/env python3
"""
Test script to demonstrate query sanitization functionality
"""

import requests
import json

# Your deployed Modal endpoint
ENDPOINT = "https://parth2sachdeva--hackrx-rag-optimized-new-enhancedmodelco-fcd5d6.modal.run"
TOKEN = "your-secret-token"  # Replace with your actual token

def test_query_sanitization():
    """Test the query sanitization feature"""
    
    # Test payload for the /hackrx/run endpoint
    payload = {
        "documents": "https://example.com/sample-document.pdf",  # Replace with actual document URL
        "questions": [
            "Hi there!",
            "What is this document about?",
            "Tell me a fun fact about this document",
            "What are the main points?",
            "How can I use this information?"
        ],
        "config": {
            "chunking": {
                "chunk_size": 512,
                "chunk_overlap": 50
            },
            "retrieval": {
                "semantic_search_top_k": 20,
                "bm25_search_top_k": 20,
                "hybrid_fusion_top_k": 10,
                "final_context_top_k": 5
            },
            "rrf": {
                "semantic_weight": 0.7,
                "bm25_weight": 0.3,
                "k_parameter": 60
            },
            "generation": {
                "max_tokens": 150,
                "temperature": 0.1
            },
            "performance": {
                "max_search_workers": 3,
                "max_generation_workers": 2,
                "reranking_batch_size": 16,
                "embedding_batch_size": 32
            }
        }
    }
    
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    
    print("🧪 Testing Query Sanitization Feature")
    print("=" * 50)
    
    try:
        # Make the request
        response = requests.post(
            f"{ENDPOINT}/hackrx/run",
            headers=headers,
            json=payload,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Request successful!")
            print(f"📊 Generated {len(result.get('answers', []))} answers")
            
            # Print answers
            for i, answer in enumerate(result.get('answers', []), 1):
                print(f"\n📝 Answer {i}: {answer}")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_upload_and_chat():
    """Test the upload + chat workflow with query sanitization"""
    
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    
    print("\n🧪 Testing Upload + Chat with Query Sanitization")
    print("=" * 50)
    
    # Step 1: Upload document
    upload_payload = {
        "document_url": "https://example.com/sample-document.pdf",  # Replace with actual document URL
        "config": {
            "chunking": {
                "chunk_size": 512,
                "chunk_overlap": 50
            },
            "retrieval": {
                "semantic_search_top_k": 20,
                "bm25_search_top_k": 20,
                "hybrid_fusion_top_k": 10,
                "final_context_top_k": 5
            },
            "rrf": {
                "semantic_weight": 0.7,
                "bm25_weight": 0.3,
                "k_parameter": 60
            },
            "generation": {
                "max_tokens": 150,
                "temperature": 0.1
            },
            "performance": {
                "max_search_workers": 3,
                "max_generation_workers": 2,
                "reranking_batch_size": 16,
                "embedding_batch_size": 32
            }
        },
        "email": {
            "enabled": True,
            "to_email": "your-email@example.com"  # Replace with your email
        }
    }
    
    try:
        # Upload document
        print("📤 Uploading document...")
        upload_response = requests.post(
            f"{ENDPOINT}/hackrx/upload",
            headers=headers,
            json=upload_payload,
            timeout=300
        )
        
        if upload_response.status_code == 200:
            upload_result = upload_response.json()
            session_id = upload_result["session_id"]
            print(f"✅ Document uploaded! Session ID: {session_id}")
            print(f"📊 Created {upload_result['chunks_count']} chunks")
            
            # Step 2: Chat with the document
            chat_payload = {
                "session_id": session_id,
                "questions": [
                    "Hello!",
                    "What is this document about?",
                    "Tell me something interesting from this document",
                    "What are the key takeaways?",
                    "How can I apply this information?"
                ],
                "config": upload_payload["config"],  # Use same config
                "email": {
                    "enabled": True,
                    "to_email": "your-email@example.com"  # Replace with your email
                }
            }
            
            print("\n💬 Chatting with document...")
            chat_response = requests.post(
                f"{ENDPOINT}/hackrx/chat",
                headers=headers,
                json=chat_payload,
                timeout=300
            )
            
            if chat_response.status_code == 200:
                chat_result = chat_response.json()
                print("✅ Chat successful!")
                print(f"📊 Generated {len(chat_result.get('answers', []))} answers")
                
                # Print answers
                for i, answer in enumerate(chat_result.get('answers', []), 1):
                    print(f"\n📝 Answer {i}: {answer}")
                    
            else:
                print(f"❌ Chat failed with status {chat_response.status_code}")
                print(f"Response: {chat_response.text}")
                
        else:
            print(f"❌ Upload failed with status {upload_response.status_code}")
            print(f"Response: {upload_response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Query Sanitization Test Suite")
    print("=" * 50)
    print("This script tests the new query sanitization feature where:")
    print("1. Original question + 5 augmented queries are sent to LLM")
    print("2. LLM returns a single refined question")
    print("3. The refined question is used for answer generation")
    print("=" * 50)
    
    # Uncomment the test you want to run:
    # test_query_sanitization()
    # test_upload_and_chat()
    
    print("\n⚠️  To run the tests:")
    print("1. Replace 'your-secret-token' with your actual token")
    print("2. Replace 'https://example.com/sample-document.pdf' with a real document URL")
    print("3. Replace 'your-email@example.com' with your email address")
    print("4. Uncomment the test functions you want to run")
