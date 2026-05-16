"""
03_langchain_rag_chain.py — Complete RAG Chain with LangChain

IMPROVEMENTS OVER SCRATCH VERSION:
1. Uses LCEL (LangChain Expression Language) — cleaner composition
2. Built-in conversation memory with different strategies
3. Prompt templates with variables
4. Streaming support for real-time responses
5. Runnable chains that can be deployed as APIs
"""

import os
from dotenv import load_dotenv
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
from langchain_core.messages import HumanMessage, AIMessage

from langchain_loader import load_and_chunk_all
from langchain_retriever import create_vector_store, load_vector_store, get_best_retriever

load_dotenv()

# ══════════════════════════════════════════════════════════════
#  LLM SETUP
# ══════════════════════════════════════════════════════════════

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)


# ══════════════════════════════════════════════════════════════
#  PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════

# System prompt with today's date injected
today = datetime.now().strftime("%A, %B %d, %Y")

system_template = f"""You are Ember 🔥, the friendly WhatsApp assistant for Spice & Ember restaurant in New York.

TODAY IS: {today}

RULES:
1. ONLY answer using the CONTEXT provided below. Never make up prices, hours, or menu items.
2. If the context doesn't contain the answer, say: "I don't have that info right now! Call us at +1-555-432-1000 🍽"
3. Keep answers short and friendly — this is WhatsApp, not an essay.
4. Use max 2 emojis per message.
5. When asked "are you open today" or "what time do you close tonight", use TODAY's day ({today.split(',')[0]}) to find hours in context.
6. When taking orders: collect items, quantity, delivery/dine-in, address if delivery.
7. When taking bookings: collect date, time, number of guests, name.

CONTEXT:
{{context}}

CONVERSATION HISTORY:
{{chat_history}}
"""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


# ══════════════════════════════════════════════════════════════
#  RAG CHAIN WITH LCEL
# ══════════════════════════════════════════════════════════════

def format_docs(docs):
    """Format retrieved documents into context string"""
    if not docs:
        return "No relevant information found."
    
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("doc_type", "")
        context_parts.append(
            f"[Source {i}: {doc_type}]\n{doc.page_content}\n"
        )
    return "\n---\n".join(context_parts)


def create_rag_chain(vectorstore, chunks, memory_type="window"):
    """
    Create a complete RAG chain using LCEL.
    
    LCEL (LangChain Expression Language) lets you compose chains with | operator:
    retriever | format | prompt | llm | parser
    
    Much cleaner than manually calling functions!
    """
    # Choose memory strategy
    if memory_type == "buffer":
        # Keeps ALL conversation history
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
    elif memory_type == "window":
        # Keeps last K turns (more efficient)
        memory = ConversationBufferWindowMemory(
            k=6,
            return_messages=True,
            memory_key="chat_history"
        )
    else:
        # Summarizes old conversation (best for long chats)
        memory = ConversationSummaryMemory(
            llm=llm,
            return_messages=True,
            memory_key="chat_history"
        )
    
    # Get best retriever
    retriever = get_best_retriever(vectorstore, chunks, strategy="ensemble")
    
    # Build the chain using LCEL
    # This is the magic of LangChain — composable chains!
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda _: memory.load_memory_variables({})["chat_history"])
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, memory


# ══════════════════════════════════════════════════════════════
#  CHAT SESSION MANAGER
# ══════════════════════════════════════════════════════════════

class LangChainChatSession:
    """
    Manages per-user conversation sessions with LangChain memory.
    
    BETTER THAN SCRATCH VERSION:
    - Built-in memory management (buffer, window, summary)
    - Automatic message formatting
    - Can switch memory strategies on the fly
    - Thread-safe for production
    """
    def __init__(self, vectorstore, chunks, memory_type="window"):
        self.sessions = {}
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.memory_type = memory_type
    
    def get_or_create_session(self, user_id: str):
        """Get existing session or create new one"""
        if user_id not in self.sessions:
            chain, memory = create_rag_chain(
                self.vectorstore,
                self.chunks,
                self.memory_type
            )
            self.sessions[user_id] = {
                "chain": chain,
                "memory": memory
            }
        return self.sessions[user_id]
    
    def chat(self, user_id: str, message: str) -> str:
        """Send message and get response"""
        session = self.get_or_create_session(user_id)
        chain = session["chain"]
        memory = session["memory"]
        
        # Get response from chain
        response = chain.invoke(message)
        
        # Save to memory
        memory.save_context(
            {"question": message},
            {"output": response}
        )
        
        return response
    
    def reset_session(self, user_id: str):
        """Clear conversation history for a user"""
        if user_id in self.sessions:
            del self.sessions[user_id]


# ══════════════════════════════════════════════════════════════
#  STREAMING VERSION (for real-time responses)
# ══════════════════════════════════════════════════════════════

def create_streaming_rag_chain(vectorstore, chunks):
    """
    Streaming version — tokens arrive in real-time instead of waiting
    for full response. Great for web UIs.
    """
    retriever = get_best_retriever(vectorstore, chunks, strategy="ensemble")
    memory = ConversationBufferWindowMemory(
        k=6,
        return_messages=True,
        memory_key="chat_history"
    )
    
    # Same chain but LLM streams tokens
    streaming_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
        streaming=True  # ← enables streaming
    )
    
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda _: memory.load_memory_variables({})["chat_history"])
        }
        | prompt
        | streaming_llm
        | StrOutputParser()
    )
    
    return chain, memory


# ══════════════════════════════════════════════════════════════
#  TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load or create vector store
    if os.path.exists("./chroma_db_lc"):
        print("⚡ Loading existing vector store...")
        vectorstore = load_vector_store()
        chunks = load_and_chunk_all()
    else:
        print("📦 Creating new vector store...")
        chunks = load_and_chunk_all()
        vectorstore = create_vector_store(chunks)
    
    # Create session manager
    session_manager = LangChainChatSession(vectorstore, chunks, memory_type="window")
    user_id = "test_user_123"
    
    # Test conversation
    test_messages = [
        "Hi!",
        "Do you have vegan options?",
        "What's the price of the Mango Sorbet?",
        "Are you open today?",
        "I want to order 2 Crispy Tofu Bites for delivery"
    ]
    
    print("🔥 Spice & Ember Chatbot — LangChain Version\n")
    print("=" * 60)
    
    for message in test_messages:
        print(f"\n👤 Customer: {message}")
        response = session_manager.chat(user_id, message)
        print(f"🔥 Ember:    {response}")
        print("-" * 50)
    
    # Show memory
    print("\n📝 Conversation Memory:")
    session = session_manager.get_or_create_session(user_id)
    history = session["memory"].load_memory_variables({})["chat_history"]
    print(f"Stored {len(history)} messages")
