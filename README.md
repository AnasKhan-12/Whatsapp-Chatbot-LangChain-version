# 🔥 Spice & Ember — AI WhatsApp Chatbot

An intelligent WhatsApp chatbot for **Spice & Ember restaurant**, powered by a Retrieval-Augmented Generation (RAG) pipeline built with LangChain and Google Gemini. Customers can ask anything — menu, reservations, hours, specials — and get instant, accurate replies on WhatsApp.

---

## 🎥 Demo

<img width="790" height="820" alt="WhatsApp Image 2026-05-17 at 1 48 40 AM" src="https://github.com/user-attachments/assets/0ff27cf7-d185-42cc-bf1f-8f5608007533" />


---

## 🧠 How It Works

```
Customer WhatsApp Message
        ↓
Meta Cloud API
        ↓
FastAPI Webhook  (/webhook)
        ↓
LangChain RAG Pipeline
   ├── Ensemble Retriever (Vector Search + BM25 Keyword Search)
   ├── Semantic Chunking
   ├── Conversation Memory (last 6 turns)
   └── Google Gemini LLM
        ↓
Reply sent back via Meta Graph API
        ↓
Customer receives answer on WhatsApp
```

---

## ✨ Features

- 💬 **Natural conversation** — remembers context across multiple messages
- 🔍 **Ensemble retrieval** — combines vector similarity + keyword search for best accuracy
- 📄 **Reads your data** — ingests PDF menus and Excel sheets automatically
- ⚡ **Semantic chunking** — splits documents by meaning, not just token count
- 🔄 **Session management** — per-user conversation history
- 🛡️ **Error handling** — graceful fallbacks for quota limits and timeouts
- 📊 **Stats endpoint** — monitor active sessions in real time

---

## 🗂️ Project Structure

```
├── langchain_loader.py          # Loads PDF + Excel, semantic chunking
├── langchain_retriever.py       # 5 retrieval strategies (ensemble, compressed, etc.)
├── langchain_rag_chain.py       # LCEL chains + conversation memory
├── langchain_whatsapp_bot.py    # FastAPI webhook + Meta Cloud API integration
├── spice_and_ember_data.pdf     # Restaurant info (hours, policies, FAQs)
├── spice_and_ember_menu.xlsx    # Full menu with prices
├── requirements.txt             # All dependencies
└── .env                         # API keys (never commit this!)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini (via `langchain-google-genai`) |
| RAG Framework | LangChain + LCEL |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (`rank-bm25`) |
| Embeddings | Sentence Transformers |
| Web Framework | FastAPI + Uvicorn |
| WhatsApp API | Meta Cloud API |
| HTTP Client | httpx |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A Meta Developer account with a WhatsApp Business App
- Google Gemini API key
- ngrok (for local testing)

---

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/spice-ember-whatsapp-bot.git
cd spice-ember-whatsapp-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your `.env` file

Create a `.env` file in the root folder:

```env
# Meta Cloud API
WHATSAPP_TOKEN=your_whatsapp_token
PHONE_NUMBER_ID=your_phone_number_id
VERIFY_TOKEN=your_custom_verify_token

# Google Gemini
GOOGLE_API_KEY=your_gemini_api_key
```

| Variable | Where to get it |
|----------|----------------|
| `WHATSAPP_TOKEN` | Meta Developer Portal → WhatsApp → API Setup |
| `PHONE_NUMBER_ID` | Meta Developer Portal → WhatsApp → API Setup |
| `VERIFY_TOKEN` | Make up any string (e.g. `spice_ember_verify_123`) |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |

### 4. Add your restaurant data

Place these files in the root folder:
- `spice_and_ember_data.pdf` — restaurant info, FAQs, policies
- `spice_and_ember_menu.xlsx` — menu items and prices

### 5. Build the vector store

```bash
python langchain_loader.py
```

This reads your PDF and Excel files, chunks them semantically, and saves the vector store to `./chroma_db_lc`.

### 6. Run the bot locally

```bash
uvicorn langchain_whatsapp_bot:app --reload --port 8000
```

### 7. Expose it publicly with ngrok

```bash
ngrok http 8000
```

Copy the `https://` URL ngrok gives you.

### 8. Register your webhook in Meta

1. Meta Developer Portal → **WhatsApp → Configuration → Webhook**
2. **Callback URL:** `https://your-ngrok-url.ngrok-free.app/webhook`
3. **Verify Token:** same string you put in `.env`
4. Click **Verify and Save**
5. Under **Webhook Fields** → subscribe to **`messages`**

### 9. Add your test number

Meta Developer Portal → **WhatsApp → API Setup** → add your personal number to the recipient list.

### 10. Send a WhatsApp message and get a reply! 🎉

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/webhook` | Meta webhook verification |
| `POST` | `/webhook` | Receives incoming WhatsApp messages |
| `GET` | `/stats` | Active sessions + vector store info |
| `POST` | `/reset/{user_id}` | Reset a user's conversation |

---

## 🌐 Deploying to Production (Render)

1. Push your repo to GitHub (make sure `.env` is in `.gitignore`!)
2. Go to [render.com](https://render.com) → New Web Service → connect your repo
3. Add all 4 environment variables in Render's dashboard
4. Start command: `uvicorn langchain_whatsapp_bot:app --host 0.0.0.0 --port 8000`
5. Once deployed, update your Meta webhook URL to the Render URL

---

## 💬 Built-in Commands

Customers can type these in WhatsApp:

| Command | Action |
|---------|--------|
| `reset` | Clears conversation history |
| `start` | Restarts the conversation |
| `restart` | Restarts the conversation |

---

## ⚠️ Important Notes

- **Never commit your `.env` file** — add it to `.gitignore`
- Meta temporary tokens expire every **24 hours** — use a System User token for production
- The free Render tier sleeps after inactivity — upgrade for production use
- Meta enforces a **24-hour messaging window** — you can only reply within 24hrs of a customer's last message

---

## 📄 License

MIT License — feel free to use and modify for your own restaurant or business.

---

## 🙌 Acknowledgements

- [LangChain](https://langchain.com) for the RAG framework
- [Google Gemini](https://aistudio.google.com) for the LLM
- [Meta Cloud API](https://developers.facebook.com/docs/whatsapp) for WhatsApp integration
- [ChromaDB](https://www.trychroma.com) for vector storage
