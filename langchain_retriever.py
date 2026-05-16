"""
02_langchain_retriever.py — Advanced Retrieval with LangChain

IMPROVEMENTS OVER SCRATCH VERSION:
1. Ensemble retriever combines vector + keyword search
2. Contextual compression removes irrelevant parts
3. Multi-query retriever generates query variations
4. Self-query retriever extracts metadata filters from natural language
5. Re-ranking for better result quality
"""

import os
from dotenv import load_dotenv
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    MultiQueryRetriever
)
from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter
)
from langchain_community.retrievers import BM25Retriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import SelfQueryRetriever
load_dotenv()

# Initialize LLM and embeddings
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# ══════════════════════════════════════════════════════════════
#  VECTOR STORE SETUP
# ══════════════════════════════════════════════════════════════

def create_vector_store(chunks: List[Document], persist_dir: str = "./chroma_db_lc"):
    """
    Create ChromaDB vector store with LangChain.
    
    BETTER THAN SCRATCH:
    - One line to embed + store (no manual batching)
    - Automatic retry on failures
    - Built-in deduplication
    """
    print(f"🔄 Creating vector store from {len(chunks)} chunks...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="spice_ember_lc"
    )
    
    print(f"✅ Vector store created with {vectorstore._collection.count()} chunks")
    return vectorstore


def load_vector_store(persist_dir: str = "./chroma_db_lc"):
    """Load existing vector store"""
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="spice_ember_lc"
    )
    return vectorstore


# ══════════════════════════════════════════════════════════════
#  ADVANCED RETRIEVERS
# ══════════════════════════════════════════════════════════════

def get_basic_retriever(vectorstore, k=4):
    """Basic vector similarity search"""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def get_ensemble_retriever(vectorstore, chunks: List[Document], k=4):
    """
    Combines vector search + keyword search.
    
    WHY THIS IS BETTER:
    - Vector search: finds semantically similar chunks
    - BM25 (keyword): finds exact matches (prices, item IDs, names)
    - Ensemble: returns best of both, weighted combination
    
    Example: "what costs $12.99" → keyword search finds exact price
             "cheap starter" → vector search finds semantically similar
    """
    # Vector retriever
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    
    # Keyword retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    
    # Combine both with 60% vector, 40% keyword weighting
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
    
    return ensemble


def get_compressed_retriever(base_retriever, compression_type="llm"):
    """
    Takes retrieved chunks and removes irrelevant sentences.
    
    WHY THIS IS BETTER:
    - Your scratch version returns entire chunks (200-300 tokens each)
    - This extracts only the 2-3 sentences that actually answer the query
    - Sends less noise to the LLM → better answers
    
    Example:
    Query: "Is the ribeye gluten-free?"
    
    Scratch version returns:
    "Ember Ribeye Steak. Aged 21 days. $38.99. 850 calories.
     Spice level: none. Vegetarian: No. Gluten-free: Yes.
     Chef's note: We dry-age in-house..."  (100 tokens)
    
    Compressed version returns:
    "Gluten-free: Yes"  (3 tokens)
    """
    if compression_type == "llm":
        # Uses LLM to extract relevant parts
        compressor = LLMChainExtractor.from_llm(llm)
    else:
        # Uses embeddings similarity to filter sentences
        compressor = EmbeddingsFilter(
            embeddings=embeddings,
            similarity_threshold=0.7
        )
    
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )


def get_multi_query_retriever(vectorstore):
    """
    Generates multiple variations of the user's query, searches with each,
    then deduplicates results.
    
    WHY THIS IS BETTER:
    User: "vegan stuff"
    Scratch version: searches "vegan stuff" → misses results
    
    Multi-query version generates:
    - "vegan menu items"
    - "plant-based options"
    - "dairy-free and egg-free dishes"
    
    Searches with all 3 → better recall
    """
    return MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )


def get_self_query_retriever(vectorstore):
    """
    Extracts metadata filters from natural language queries.
    
    WHY THIS IS SMARTER THAN SCRATCH VERSION:
    
    Scratch version:
    - You manually wrote intent detection keywords
    - If user says "spicy main courses" your code doesn't catch "main courses"
    
    Self-query version:
    User: "show me spicy main courses under $20"
    LLM automatically extracts:
      query = "spicy"
      filter = {
          "doc_type": "menu_item",
          "category": "main",
          "price": {"$lt": 20}
      }
    
    No manual keyword lists needed — LLM figures it out!
    """
    metadata_field_info = [
        AttributeInfo(
            name="doc_type",
            description="Type of document: menu_item, hours, faq, chef_note, restaurant_info",
            type="string"
        ),
        AttributeInfo(
            name="category",
            description="Menu category: starter, main, dessert, drink",
            type="string"
        ),
        AttributeInfo(
            name="is_vegetarian",
            description="Whether the item is vegetarian: Yes or No",
            type="string"
        ),
        AttributeInfo(
            name="is_vegan",
            description="Whether the item is vegan: Yes or No",
            type="string"
        ),
        AttributeInfo(
            name="spice_level",
            description="Spice level: no spice, mild, medium, hot, very hot",
            type="string"
        ),
        AttributeInfo(
            name="day",
            description="Day of the week for opening hours",
            type="string"
        )
    ]
    
    document_content_description = "Restaurant menu items, hours, FAQs, and information"
    
    return SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True
    )


# ══════════════════════════════════════════════════════════════
#  SMART RETRIEVER SELECTOR
# ══════════════════════════════════════════════════════════════

def get_best_retriever(
    vectorstore,
    chunks: List[Document],
    strategy: str = "ensemble_compressed"
):
    """
    Returns the best retriever for the use case.
    
    Strategies:
    - basic: simple vector search (fastest)
    - ensemble: vector + keyword (best balance)
    - compressed: ensemble + compression (highest quality)
    - multi_query: query expansion (best recall)
    - self_query: natural language filters (smartest)
    """
    if strategy == "basic":
        return get_basic_retriever(vectorstore)
    
    elif strategy == "ensemble":
        return get_ensemble_retriever(vectorstore, chunks)
    
    elif strategy == "ensemble_compressed":
        ensemble = get_ensemble_retriever(vectorstore, chunks)
        return get_compressed_retriever(ensemble, compression_type="embeddings")
    
    elif strategy == "multi_query":
        return get_multi_query_retriever(vectorstore)
    
    elif strategy == "self_query":
        return get_self_query_retriever(vectorstore)
    
    else:
        return get_ensemble_retriever(vectorstore, chunks)


# ══════════════════════════════════════════════════════════════
#  TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from langchain_loader import load_and_chunk_all
    
    # Load or create vector store
    if os.path.exists("./chroma_db_lc"):
        print("⚡ Loading existing vector store...")
        vectorstore = load_vector_store()
        
        # Need chunks for BM25 — reload them
        chunks = load_and_chunk_all()
    else:
        print("📦 Creating new vector store...")
        chunks = load_and_chunk_all()
        vectorstore = create_vector_store(chunks)
    
    # Test different retrievers
    queries = [
        "Do you have vegan options?",
        "What costs less than $15?",
        "Tell me about the ribeye steak",
        "Are you open on Monday?"
    ]
    
    strategies = ["basic", "ensemble", "ensemble_compressed", "self_query"]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy}")
        print('='*60)
        
        try:
            retriever = get_best_retriever(vectorstore, chunks, strategy=strategy)
            
            for query in queries[:2]:  # test first 2 queries
                print(f"\nQuery: {query}")
                results = retriever.invoke(query)
                print(f"Retrieved {len(results)} chunks")
                if results:
                    print(f"Top result: {results[0].page_content[:150]}...")
        except Exception as e:
            print(f"Error with {strategy}: {e}")
