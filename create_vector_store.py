import os
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import traceback
from tqdm import tqdm  # For progress bars

# --- Configuration ---
PDF_FILE_PATH = r"CI-16000.74-International-Conventions.pdf"  # Replace with your PDF file path
CHROMA_DB_PATH = "rag_chroma_db"  # Directory to store ChromaDB
CHUNK_SIZE = 500  # Adjust as needed
CHUNK_OVERLAP = 100  # Adjust as needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Langchain embedding model, approx 512 dimensions

def extract_text_with_fallback(pdf_path):
    """Extract text from PDF with multiple fallback methods"""
    all_text = []
    
    try:
        # Try opening the PDF
        pdf_document = fitz.open(pdf_path)
        total_pages = pdf_document.page_count
        print(f"PDF has {total_pages} pages")
        
        # Process each page with a progress bar
        for page_num in tqdm(range(total_pages), desc="Processing PDF pages"):
            try:
                # Method 1: Basic text extraction
                page = pdf_document[page_num]
                text = ""
                
                try:
                    # Try the standard method first
                    text = page.get_text()
                except Exception:
                    pass
                
                # If that fails or returns empty, try alternate methods
                if not text:
                    try:
                        # Method 2: Try with different parameters
                        text = page.get_text("text")
                    except Exception:
                        pass
                
                # If still empty, try blocks extraction
                if not text:
                    try:
                        # Method 3: Extract text blocks
                        blocks = page.get_text("blocks")
                        text = "\n".join([b[4] for b in blocks if isinstance(b, tuple) and len(b) > 4])
                    except Exception:
                        pass
                
                # If text is still empty after all methods, log it
                if not text.strip():
                    print(f"Warning: Page {page_num + 1} appears to be empty or unreadable")
                    continue
                
                all_text.append(text)
                
            except Exception as e:
                print(f"Error processing page {page_num + 1}: {str(e)}")
                traceback.print_exc()
                continue
        
        pdf_document.close()
        
    except Exception as e:
        print(f"Error opening PDF: {str(e)}")
        traceback.print_exc()
        return []
    
    return all_text

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Splits text into chunks with overlap."""
    chunks = []
    start_index = 0
    text_length = len(text)
    
    # Skip empty text
    if not text.strip():
        return chunks
    
    print("Text length is: ",len(text.strip()))
    while start_index < text_length:
        end_index = start_index + chunk_size
        if end_index<text_length:
            chunk = text[start_index:end_index]
        else:
            chunk = text[start_index:]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)
        
        if (end_index-chunk_overlap)<start_index:
            start_index=+1
            print(start_index,end_index,text_length)
        else:
            start_index = end_index - chunk_overlap
        
    return chunks

def create_embeddings(chunks, embedding_model_name=EMBEDDING_MODEL):
    """Generates embeddings for text chunks using Langchain HuggingFaceEmbeddings."""
    if not chunks:
        print("No chunks to create embeddings for!")
        return []
        
    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Process in batches to avoid potential memory issues
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            batch_embeddings = embeddings_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            print(f"Processed embedding batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        except Exception as e:
            print(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
            # Continue with the next batch
    
    return all_embeddings

def store_in_chromadb(chunks, embeddings, db_path=CHROMA_DB_PATH):
    """Stores text chunks and embeddings in ChromaDB."""
    if not chunks or not embeddings:
        print("No chunks or embeddings to store!")
        return
    
    if len(chunks) != len(embeddings):
        print(f"Warning: Number of chunks ({len(chunks)}) doesn't match number of embeddings ({len(embeddings)})")
        # Use the smaller number
        count = min(len(chunks), len(embeddings))
        chunks = chunks[:count]
        embeddings = embeddings[:count]
        
    # Ensure the directory exists
    os.makedirs(db_path, exist_ok=True)
    
    try:
        client = chromadb.PersistentClient(path=db_path)
        
        # Check if collection exists and recreate if needed
        try:
            # Try to get the collection
            collection = client.get_collection("rag_collection")
            print("Found existing collection. Deleting and recreating...")
            client.delete_collection("rag_collection")
        except:
            # Collection doesn't exist yet
            pass
        
        # Create a new collection
        collection = client.create_collection("rag_collection")

        # Generate unique IDs
        ids = [str(i) for i in range(len(chunks))]
        
        # Store in batches to handle larger datasets
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            batch_ids = ids[i:end_idx]
            batch_chunks = chunks[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_chunks
            )
            print(f"Stored batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} in ChromaDB")

        print(f"Successfully stored {len(chunks)} chunks in ChromaDB at '{db_path}'")
    except Exception as e:
        print(f"Error storing data in ChromaDB: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":

    # Extract text from PDF using the new method
    print(f"Processing PDF: {PDF_FILE_PATH}")
    page_texts = extract_text_with_fallback(PDF_FILE_PATH)
    
    if not page_texts:
        print("Failed to extract any text from the PDF.")
        exit(1)
    
    print(f"Successfully extracted text from {len(page_texts)} pages")
    
    # Chunk the extracted text
    all_chunks = []
    for i, text in enumerate(page_texts):
        if i==0:
            continue #skip first page, which is title.
        print("Making chunks from page",i+1)
        page_chunks = chunk_text(text)
        all_chunks.extend(page_chunks)
        print(f"Page {i+1}: Created {len(page_chunks)} chunks")
    
    print(f"Created {len(all_chunks)} chunks in total")
    
    if all_chunks:
        # Create embeddings
        embeddings = create_embeddings(all_chunks)
        
        if embeddings:
            # Store in database
            store_in_chromadb(all_chunks, embeddings)
            print("RAG database built successfully!")
        else:
            print("Failed to create embeddings.")
    else:
        print("No chunks were created.")
