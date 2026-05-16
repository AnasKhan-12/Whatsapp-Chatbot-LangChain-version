"""
04_langchain_whatsapp_bot.py — WhatsApp Bot with LangChain RAG
Integrated with Meta Cloud API (not Twilio)
"""

import os
import httpx
from dotenv import load_dotenv
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

from langchain_loader import load_and_chunk_all
from langchain_retriever import create_vector_store, load_vector_store
from langchain_rag_chain import LangChainChatSession

load_dotenv()

# ══════════════════════════════════════════════════════════════
#  STARTUP — Build Vector Store if Needed
# ══════════════════════════════════════════════════════════════

print("🚀 Starting Spice & Ember WhatsApp Bot (LangChain Version)...")

if os.path.exists("./chroma_db_lc"):
    print("⚡ Loading existing vector store...")
    vectorstore = load_vector_store()
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

session_manager = LangChainChatSession(
    vectorstore,
    chunks,
    memory_type="window"
)

print("✅ Bot ready!")


# ══════════════════════════════════════════════════════════════
#  META CLOUD API — Send Reply
# ══════════════════════════════════════════════════════════════

async def send_whatsapp_reply(to: str, message: str):
    """
    Sends a reply back to the user via Meta Cloud API.
    CHANGED: Replaced Twilio TwiML response with a direct POST
    to Meta's graph API. Meta doesn't use TwiML — it expects
    a separate outbound HTTP call to /messages.
    """
    url = f"https://graph.facebook.com/v19.0/{os.getenv('PHONE_NUMBER_ID')}/messages"
    headers = {
        "Authorization": f"Bearer {os.getenv('WHATSAPP_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()


# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="Spice & Ember Bot (LangChain + Meta Cloud API)")


@app.get("/")
async def health_check():
    return {
        "status": "running",
        "bot": "Spice & Ember WhatsApp Bot 🔥 (LangChain Version)",
        "webhook": "/webhook",
        "features": [
            "Ensemble retriever (vector + keyword search)",
            "Contextual compression",
            "Conversation memory (window strategy)",
            "Semantic chunking",
            "Meta Cloud API integration"
        ],
        "message": "Bot is live and ready!",
        "timestamp": datetime.now().isoformat()
    }


# ══════════════════════════════════════════════════════════════
#  WEBHOOK — Verification (GET) + Incoming Messages (POST)
# ══════════════════════════════════════════════════════════════

@app.get("/webhook")
def verify_webhook(request: Request):
    """
    Meta calls this once when you register your webhook URL.
    It sends hub.mode, hub.verify_token, hub.challenge.
    We must echo back hub.challenge if the token matches.

    CHANGED: Added this endpoint. Original file had it but also
    imported HTTPException without using it — fixed that too.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == os.getenv("VERIFY_TOKEN"):
        print("✅ Webhook verified by Meta!")
        return PlainTextResponse(challenge)

    print("❌ Webhook verification failed — token mismatch")
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def receive_message(request: Request):
    """
    Meta posts every incoming WhatsApp message here as JSON.

    CHANGED (major):
    - Removed Twilio Form(...) parameters and TwiML response entirely.
      Twilio uses form-encoded POST + TwiML XML reply.
      Meta uses JSON POST + separate outbound API call to send reply.
    - Removed duplicate @app.post("/webhook") route (was defined twice).
    - Properly extracts sender phone + message text from Meta's payload
      structure: data.entry[0].changes[0].value.messages[0]
    - Calls send_whatsapp_reply() instead of returning TwiML.
    - Returns 200 OK immediately (Meta requires this — if you don't
      respond with 200 fast, it retries the webhook repeatedly).
    """
    data = await request.json()

    # Meta wraps everything in entry → changes → value
    try:
        entry = data["entry"][0]
        change = entry["changes"][0]["value"]

        # Ignore non-message events (status updates, read receipts, etc.)
        if "messages" not in change:
            return {"status": "ignored"}

        msg = change["messages"][0]
        user_phone = msg["from"]          # e.g. "923001234567"
        message_type = msg.get("type")

        # Only handle text messages for now
        if message_type != "text":
            await send_whatsapp_reply(
                user_phone,
                "Sorry, I can only handle text messages right now! 🙏"
            )
            return {"status": "ok"}

        message = msg["text"]["body"].strip()

    except (KeyError, IndexError):
        # Malformed payload — return 200 anyway so Meta doesn't retry
        return {"status": "ok"}

    print(f"\n📱 Message from {user_phone}: {message}")

    # Handle reset commands
    if message.lower() in ["reset", "start", "restart"]:
        session_manager.reset_session(user_phone)
        response_text = "Hi! 👋 I'm Ember, your assistant for Spice & Ember restaurant. How can I help you today? 🔥"
    else:
        try:
            response_text = session_manager.chat(user_phone, message)
        except Exception as e:
            print(f"❌ Error: {e}")
            if "429" in str(e) or "quota" in str(e).lower():
                response_text = "Sorry, I'm a bit busy right now! Try again in a few minutes 🙏"
            elif "timeout" in str(e).lower():
                response_text = "Hmm, that's taking longer than expected. Can you try again? 🤔"
            else:
                response_text = "Oops, something went wrong! Call us at +1-555-432-1000 🍽"

    print(f"🔥 Ember: {response_text}")

    # Send reply via Meta Cloud API
    await send_whatsapp_reply(user_phone, response_text)

    # Always return 200 to Meta
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════════
#  STATS + RESET ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/stats")
async def get_stats():
    return {
        "active_sessions": len(session_manager.sessions),
        "session_ids": list(session_manager.sessions.keys()),
        "vector_store_size": vectorstore._collection.count(),
        "chunks_loaded": len(chunks)
    }


@app.post("/reset/{user_id}")
async def reset_user_session(user_id: str):
    session_manager.reset_session(user_id)
    return {"status": "success", "message": f"Session reset for {user_id}"}


# ══════════════════════════════════════════════════════════════
#  RUN SERVER
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print("\n🔥 Spice & Ember WhatsApp Bot — LangChain + Meta Cloud API")
    print("   Webhook URL: http://localhost:8000/webhook")
    print("   Stats:       http://localhost:8000/stats\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")