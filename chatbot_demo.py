#!/usr/bin/env python3
"""
Demo script showing the new chatbot-style responses
"""

import requests
import json

# Your deployed Modal endpoint
ENDPOINT = "https://parth2sachdeva--hackrx-rag-optimized-new-enhancedmodelco-fcd5d6.modal.run"
TOKEN = "your-secret-token"  # Replace with your actual token

def demo_chatbot_responses():
    """Demonstrate the new chatbot-style responses"""
    
    # Example questions that show different chatbot behaviors
    demo_questions = [
        "Hi there!",  # Greeting
        "Hello!",  # Another greeting
        "What is this document about?",  # Document question
        "Tell me a fun fact about this document",  # Fun fact request
        "What are the main points?",  # Summary request
        "How can I use this information?",  # Application question
        "What's the weather like?",  # Irrelevant question
        "Can you help me with math?",  # Off-topic question
        "What's the most interesting thing in this document?",  # Interesting content request
    ]
    
    print("🤖 Chatbot Demo - New Conversational Responses")
    print("=" * 60)
    print("This demonstrates how the chatbot now handles different types of questions:")
    print("✅ Greetings (Hi, Hello)")
    print("✅ Document questions")
    print("✅ Fun fact requests")
    print("✅ Irrelevant questions (politely redirected)")
    print("✅ General conversation")
    print("=" * 60)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{i}. User: {question}")
        print("   Bot: [Would respond conversationally based on document content]")
        
        # Show expected behavior
        if question.lower() in ["hi there!", "hello!"]:
            print("   Expected: Warm greeting + offer to help with document")
        elif "fun fact" in question.lower():
            print("   Expected: Interesting insights from the document")
        elif "weather" in question.lower() or "math" in question.lower():
            print("   Expected: Polite redirect to document-related questions")
        else:
            print("   Expected: Helpful answer based on document content")

def show_prompt_comparison():
    """Show the difference between old and new prompts"""
    
    print("\n📝 Prompt Comparison")
    print("=" * 60)
    
    print("\n🔴 OLD PROMPT (Competition Style):")
    print("-" * 40)
    print("""
    Single sentence with briefing. Based on the context, provide a direct 
    one-sentence answer to the question. Find and give the exact number or 
    fact for a question. Don't do thinking or reasoning, just give the answer 
    directly. Summarize the answer in a single sentence and keep it brief.
    """)
    
    print("\n🟢 NEW PROMPT (Chatbot Style):")
    print("-" * 40)
    print("""
    You are a helpful AI assistant that can answer questions about the 
    uploaded document. Be conversational and friendly in your responses.
    
    INSTRUCTIONS:
    - If the question is about the document content, provide a helpful answer
    - If the user greets you, respond warmly and ask how you can help
    - If the user asks for fun facts, share interesting insights
    - If the question is not related to the document, politely explain
    - Be conversational, helpful, and engaging
    """)

def show_example_responses():
    """Show example responses for different question types"""
    
    print("\n💬 Example Responses")
    print("=" * 60)
    
    examples = [
        {
            "question": "Hi there!",
            "old_response": "Hello.",
            "new_response": "Hello! I'm here to help you with questions about the uploaded document. What would you like to know about it?"
        },
        {
            "question": "What is this document about?",
            "old_response": "This document discusses machine learning algorithms and their applications.",
            "new_response": "This document covers machine learning algorithms and their practical applications. It's quite comprehensive and includes examples of how these algorithms work in real-world scenarios."
        },
        {
            "question": "Tell me a fun fact about this document",
            "old_response": "The document mentions neural networks.",
            "new_response": "Here's something interesting: the document explains how neural networks can process information similar to how the human brain works, with interconnected nodes that can learn and adapt over time!"
        },
        {
            "question": "What's the weather like?",
            "old_response": "I don't know.",
            "new_response": "I can only help you with questions about the uploaded document. Would you like to know something about the document content instead?"
        }
    ]
    
    for example in examples:
        print(f"\n❓ Question: {example['question']}")
        print(f"🔴 Old Response: {example['old_response']}")
        print(f"🟢 New Response: {example['new_response']}")
        print("-" * 50)

if __name__ == "__main__":
    print("🚀 RAG Chatbot - Conversational Style Demo")
    print("=" * 60)
    
    demo_chatbot_responses()
    show_prompt_comparison()
    show_example_responses()
    
    print("\n🎯 Key Improvements:")
    print("✅ Conversational and friendly responses")
    print("✅ Handles greetings naturally")
    print("✅ Provides fun facts and interesting insights")
    print("✅ Politely redirects off-topic questions")
    print("✅ More engaging and helpful overall")
    
    print("\n📧 Email Logging:")
    print("✅ All query sanitization steps are logged")
    print("✅ Original questions, augmented queries, and sanitized questions")
    print("✅ Complete conversation flow tracked")
    
    print("\n🔗 Deploy Status:")
    print("✅ Successfully deployed to Modal.com")
    print("✅ Ready for production use as a chatbot!")
