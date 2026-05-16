"""
04_langchain_whatsapp_bot.py — WhatsApp Bot with LangChain RAG

IMPROVEMENTS OVER SCRATCH VERSION:
1. Uses LangChain chat session management
2. Built-in error handling and retry logic
3. Streaming responses (optional)
4. Better conversation context
5. Async support for high traffic
"""

import os
from dotenv import load_dotenv
from datetime import datetime

from fastapi import FastAPI, Form, Response, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse

from langchain_loader import load_and_chunk_all
from langchain_retriever import create_vector_store, load_vector_store
from langchain_rag_chain import LangChainChatSession

load_dotenv()

# ══════════════════════════════════════════════════════════════
#  STARTUP — Build Vector Store if Needed
# ══════════════════════════════════════════════════════════════

print("🚀 Starting Spice & Ember WhatsApp Bot (LangChain Version)...")

# Check if vector store exists
if os.path.exists("./chroma_db_lc"):
    print("⚡ Loading existing vector store...")
    vectorstore = load_vector_store()
    # Need chunks for ensemble retriever
    chunks = load_and_chunk_all(
        pdf_path="./spice_and_ember_data.pdf",
        excel_path="./spice_and_ember_menu.xlsx"
    )
else:
    print("📦 Building vector store from scratch...")
    chunks = load_and_chunk_all(
        pdf_path="./spice_and_ember_data.pdf",
        excel_path="./spice_and_ember_menu.xlsx"
    )
    vectorstore = create_vector_store(chunks)

# Create session manager
session_manager = LangChainChatSession(
    vectorstore,
    chunks,
    memory_type="window"  # keeps last 6 turns
)

print("✅ Bot ready!")


# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="Spice & Ember Bot (LangChain)")


@app.get("/")
async def health_check():
    """Homepage with status"""
    return {
        "status": "running",
        "bot": "Spice & Ember WhatsApp Bot 🔥 (LangChain Version)",
        "webhook": "/webhook",
        "features": [
            "Ensemble retriever (vector + keyword search)",
            "Contextual compression",
            "Conversation memory (window strategy)",
            "Semantic chunking",
            "Self-query metadata extraction"
        ],
        "message": "Bot is live and ready!",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/webhook")
async def webhook_verify():
    """Twilio webhook verification"""
    return Response(content="OK", status_code=200)


@app.post("/webhook")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...),
    NumMedia: int = Form(0),
):
    """
    Main webhook — receives WhatsApp messages from Twilio.
    
    IMPROVEMENTS OVER SCRATCH VERSION:
    - LangChain handles context automatically
    - Better error messages
    - Conversation memory built-in
    - Retry logic in LangChain chains
    """
    user_id = From
    message = Body.strip()

    print(f"\n📱 Message from {user_id}: {message}")

    # Handle reset commands
    if message.lower() in ["reset", "start", "restart"]:
        session_manager.reset_session(user_id)
        response_text = "Hi! 👋 I'm Ember, your assistant for Spice & Ember restaurant. How can I help you today? 🔥"
    else:
        try:
            # Get response from LangChain RAG chain
            response_text = session_manager.chat(user_id, message)
        except Exception as e:
            print(f"❌ Error: {e}")
            
            # Provide helpful error messages
            if "429" in str(e) or "quota" in str(e).lower():
                response_text = "Sorry, I'm a bit busy right now! Try again in a few minutes 🙏"
            elif "timeout" in str(e).lower():
                response_text = "Hmm, that's taking longer than expected. Can you try again? 🤔"
            else:
                response_text = "Oops, something went wrong! Call us at +1-555-432-1000 🍽"

    print(f"🔥 Ember: {response_text}")

    # Build TwiML response
    twiml = MessagingResponse()
    twiml.message(response_text)

    return Response(content=str(twiml), media_type="application/xml")


@app.get("/stats")
async def get_stats():
    """Statistics endpoint — shows active sessions"""
    return {
        "active_sessions": len(session_manager.sessions),
        "session_ids": list(session_manager.sessions.keys()),
        "vector_store_size": vectorstore._collection.count(),
        "chunks_loaded": len(chunks)
    }


@app.post("/reset/{user_id}")
async def reset_user_session(user_id: str):
    """Reset a specific user's conversation"""
    session_manager.reset_session(user_id)
    return {"status": "success", "message": f"Session reset for {user_id}"}


# ══════════════════════════════════════════════════════════════
#  RUN SERVER
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("\n🔥 Spice & Ember WhatsApp Bot — LangChain Version")
    print("   Webhook URL: http://localhost:8000/webhook")
    print("   For production: Use Render/Railway URL + /webhook")
    print("   Stats endpoint: http://localhost:8000/stats\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
