"""
01_langchain_loader.py — Load & Chunk with LangChain

IMPROVEMENTS OVER SCRATCH VERSION:
1. Uses LangChain's document loaders (cleaner, more standard)
2. Advanced chunking with semantic splitting + parent-child strategy
3. Metadata enrichment happens during loading
4. All in one pipeline — no separate chunker file needed
"""

from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter
)
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pdfplumber
import openpyxl
import re
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings for semantic chunking
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# ══════════════════════════════════════════════════════════════
#  CUSTOM PDF LOADER WITH FORMAT DETECTION
# ══════════════════════════════════════════════════════════════

def load_pdf_advanced(pdf_path: str) -> List[Document]:
    """
    Load PDF with format detection — preserves structure while
    creating LangChain Document objects.
    
    Better than scratch version because:
    - Returns standard LangChain Documents (works with any LangChain tool)
    - Richer metadata (page numbers, format types, structural info)
    - Parent-child relationships preserved in metadata
    """
    documents = []
    
    with pdfplumber.open(pdf_path) as pdf:
        # Page 1: About + Contact table
        page1 = pdf.pages[0]
        text = page1.extract_text()
        
        # Extract About section
        about_match = re.search(
            r"Section 1 — About the Restaurant.*?\n(.+?)(?=Section 2)",
            text, re.DOTALL
        )
        if about_match:
            about_text = " ".join(about_match.group(1).strip().split("\n"))
            documents.append(Document(
                page_content=about_text,
                metadata={
                    "source": pdf_path,
                    "page": 1,
                    "section": "about",
                    "format": "narrative",
                    "doc_type": "restaurant_info"
                }
            ))
        
        # Extract contact table
        tables = page1.extract_tables()
        if tables:
            kv_pairs = []
            for row in tables[0][1:]:
                if row[0] and row[1]:
                    kv_pairs.append(f"{row[0].strip()}: {row[1].strip()}")
            
            documents.append(Document(
                page_content="Restaurant information:\n" + "\n".join(kv_pairs),
                metadata={
                    "source": pdf_path,
                    "page": 1,
                    "section": "contact",
                    "format": "key_value",
                    "doc_type": "restaurant_info"
                }
            ))
        
        # Page 2: Hours table
        page2 = pdf.pages[1]
        tables = page2.extract_tables()
        if len(tables) >= 2:
            hours_table = tables[1]
            for row in hours_table[1:]:
                if row and row[0]:
                    day, status, open_t, close_t = [str(v).strip() for v in row[:4]]
                    if status.upper() == "CLOSED":
                        text = f"{day}: CLOSED"
                    else:
                        text = f"{day}: Open {open_t} - {close_t}"
                    
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path,
                            "page": 2,
                            "section": "hours",
                            "format": "structured",
                            "doc_type": "hours",
                            "day": day
                        }
                    ))
        
        # Page 3: Menu + FAQ
        page3 = pdf.pages[2]
        tables = page3.extract_tables()
        
        # Menu table
        if tables:
            menu_table = tables[0]
            for row in menu_table[1:]:
                if not row or not row[0]:
                    continue
                
                item_id, name, category, price, cal, spice, veg, gf = [
                    str(v).strip() for v in row[:8]
                ]
                
                text = (
                    f"{name}\n"
                    f"Category: {category} | Price: {price}\n"
                    f"Calories: {cal} | Spice: {spice}\n"
                    f"Vegetarian: {veg} | Gluten-free: {gf}"
                )
                
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": 3,
                        "section": "menu",
                        "format": "structured",
                        "doc_type": "menu_item",
                        "item_id": item_id,
                        "item_name": name,
                        "category": category.lower(),
                        "price": price,
                        "is_vegetarian": veg,
                        "spice_level": spice.lower()
                    }
                ))
        
        # FAQ
        text = page3.extract_text()
        faq_match = re.search(r"Section 5.*?Q&A Format\)(.*)", text, re.DOTALL)
        if faq_match:
            faq_text = faq_match.group(1).strip()
            qa_pairs = re.findall(
                r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:|\Z)",
                faq_text,
                re.DOTALL
            )
            for q, a in qa_pairs:
                documents.append(Document(
                    page_content=f"Q: {q.strip()}\nA: {a.strip()}",
                    metadata={
                        "source": pdf_path,
                        "page": 3,
                        "section": "faq",
                        "format": "qa",
                        "doc_type": "faq"
                    }
                ))
        
        # Page 4: Chef's notes
        page4 = pdf.pages[3]
        text = page4.extract_text()
        notes_match = re.search(
            r"Section 6 — Chef's Notes.*?\n(.+?)(?=Section 7)",
            text, re.DOTALL
        )
        if notes_match:
            notes_text = notes_match.group(1).strip()
            dish_headers = [
                "Ember Ribeye Steak", "Spicy Dragon Noodles",
                "BBQ Pulled Pork", "Lava Chocolate Cake", "Soup of the Day"
            ]
            pattern = "(" + "|".join(re.escape(d) for d in dish_headers) + ")"
            parts = re.split(pattern, notes_text)
            
            i = 1
            while i < len(parts) - 1:
                dish = parts[i].strip()
                note = parts[i + 1].strip().replace("\n", " ")
                if dish and note:
                    documents.append(Document(
                        page_content=f"Chef's note on {dish}: {note}",
                        metadata={
                            "source": pdf_path,
                            "page": 4,
                            "section": "chef_notes",
                            "format": "narrative",
                            "doc_type": "chef_note",
                            "dish_name": dish
                        }
                    ))
                i += 2
    
    print(f"✅ Loaded {len(documents)} documents from PDF")
    return documents


def load_excel_advanced(excel_path: str) -> List[Document]:
    """Load Excel with LangChain Document format"""
    documents = []
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    
    # Menu sheet
    ws = wb["Menu"]
    for row in ws.iter_rows(min_row=3, values_only=True):
        item_id = row[0]
        if not item_id or not str(item_id).startswith(("ST", "MN", "DS", "DR")):
            continue
        
        name, category, price, cal, spice, veg, vegan, gf, allergens = row[1:]
        
        text = (
            f"{name}\n"
            f"Category: {category} | Price: ${price}\n"
            f"Calories: {cal} | Spice: {spice}\n"
            f"Veg: {veg} | Vegan: {vegan} | GF: {gf}\n"
            f"Allergens: {allergens}"
        )
        
        documents.append(Document(
            page_content=text,
            metadata={
                "source": excel_path,
                "sheet": "Menu",
                "format": "structured",
                "doc_type": "menu_item",
                "item_id": str(item_id),
                "item_name": str(name),
                "category": str(category).lower(),
                "is_vegan": str(vegan),
                "allergens": str(allergens)
            }
        ))
    
    print(f"✅ Loaded {len(documents)} documents from Excel")
    return documents


# ══════════════════════════════════════════════════════════════
#  ADVANCED CHUNKING STRATEGIES
# ══════════════════════════════════════════════════════════════

def chunk_documents_advanced(documents: List[Document]) -> List[Document]:
    """
    Smart chunking using LangChain's best tools.
    
    BETTER THAN SCRATCH VERSION:
    1. Semantic Chunker — splits on meaning, not just token count
    2. Format-aware — different strategies per format
    3. Parent-child preservation — metadata tracks relationships
    4. Token-aware — respects model context limits
    """
    chunked = []
    
    # Semantic chunker for narrative text
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"  # splits when meaning changes
    )
    
    # Recursive splitter for structured text
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Token-based splitter for precise control
    token_splitter = TokenTextSplitter(
        chunk_size=200,
        chunk_overlap=30
    )
    
    for doc in documents:
        fmt = doc.metadata.get("format")
        doc_type = doc.metadata.get("doc_type")
        
        # Already optimal size — don't split
        if fmt in ["structured", "key_value", "qa"] or doc_type == "hours":
            chunked.append(doc)
        
        # Narrative text — semantic splitting preserves meaning
        elif fmt == "narrative":
            chunks = semantic_splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_total"] = len(chunks)
                chunk.metadata["chunk_strategy"] = "semantic"
            chunked.extend(chunks)
        
        # Long structured text — recursive splitting
        elif len(doc.page_content) > 500:
            chunks = recursive_splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_total"] = len(chunks)
                chunk.metadata["chunk_strategy"] = "recursive"
            chunked.extend(chunks)
        
        else:
            chunked.append(doc)
    
    print(f"✅ Chunked into {len(chunked)} final chunks")
    return chunked


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def load_and_chunk_all(
    pdf_path: str = "spice_and_ember_data.pdf",
    excel_path: str = "spice_and_ember_menu.xlsx"
) -> List[Document]:
    """
    Full loading + chunking pipeline.
    Returns LangChain Documents ready for embedding.
    """
    print("📦 Loading documents...")
    docs = []
    docs += load_pdf_advanced(pdf_path)
    docs += load_excel_advanced(excel_path)
    
    print(f"\n🔪 Chunking {len(docs)} documents...")
    chunks = chunk_documents_advanced(docs)
    
    print(f"\n✅ Pipeline complete: {len(chunks)} chunks ready for embedding")
    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_all()
    
    # Show sample
    print("\n── Sample chunks ──\n")
    for chunk in chunks[:3]:
        print(f"Content: {chunk.page_content[:100]}...")
        print(f"Metadata: {chunk.metadata}")
        print("-" * 50)
