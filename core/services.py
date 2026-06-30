import fitz  
import faiss
import numpy as np
import os
from django.conf import settings
from .models import Document, Chunk
from sentence_transformers import SentenceTransformer
import hashlib

def extract_text_from_pdf(document: Document) -> str:
    document.processing_status = Document.ProcessingStatus.PROCESSING
    document.save(update_fields=['processing_status'])

    try:
        pdf_path = document.file.path
        doc = fitz.open(pdf_path)
        
        extracted_pages = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text") 
            extracted_pages.append(page_text)

        full_text = "\n--- PAGE BREAK ---\n".join(extracted_pages)
        return full_text

    except Exception as e:
        document.processing_status = Document.ProcessingStatus.FAILED
        document.save(update_fields=['processing_status'])
        raise RuntimeError(f"Failed to extract text from {document.title}: {str(e)}")
    


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:

    words = text.split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        
        chunks.append(" ".join(chunk_words))
        
        start += (chunk_size - overlap)
        
        if chunk_size <= overlap:
            break
            
    return chunks

def process_and_save_chunks(document: Document, raw_text: str) -> int:
    document.chunks.all().delete()
    
    text_fragments = chunk_text(raw_text, chunk_size=600, overlap=100)
    chunk_objects = [
        Chunk(
            document=document,
            chunk_index=idx,
            original_text=fragment
        )
        for idx, fragment in enumerate(text_fragments)
    ]
    
    Chunk.objects.bulk_create(chunk_objects)
    return len(chunk_objects)


_model = None
def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def generate_embeddings_for_document(document: Document) -> int:

    chunks = document.chunks.filter(embedding__isnull=True)
    if not chunks.exists():
        return 0

    model = get_embedding_model()
    texts = [chunk.original_text for chunk in chunks]
    
    vectors = model.encode(texts, show_progress_bar=False)
    
    updated_count = 0
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector.tolist()
        chunk.save(update_fields=['embedding'])
        updated_count += 1
        
    document.processing_status = Document.ProcessingStatus.COMPLETED
    document.save(update_fields=['processing_status'])
    
    return updated_count


FAISS_INDEX_PATH = os.path.join(settings.BASE_DIR, 'faiss_index.bin')

def build_faiss_index():
    chunks = Chunk.objects.exclude(embedding__isnull=True)
    if not chunks.exists():
        return 0

    dimension = 384
    
    chunk_ids = []
    vectors = []
    
    for chunk in chunks:
        chunk_ids.append(chunk.id)
        vectors.append(chunk.embedding)
        
    vectors_np = np.array(vectors).astype('float32')
    ids_np = np.array(chunk_ids).astype('int64')

    base_index = faiss.IndexFlatL2(dimension)

    index_with_ids = faiss.IndexIDMap(base_index)

    index_with_ids.add_with_ids(vectors_np, ids_np)

    faiss.write_index(index_with_ids, FAISS_INDEX_PATH)
    
    return index_with_ids.ntotal


def retrieve_context_and_build_prompt(query: str, top_k: int = 3) -> tuple[str, str]:

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError("FAISS index not found. Please build it first.")
    
    index = faiss.read_index(FAISS_INDEX_PATH)

    model = get_embedding_model()
    query_vector = model.encode([query], show_progress_bar=False).astype('float32')

    distances, indices = index.search(query_vector, top_k)
    
    matched_ids = indices[0].tolist()

    chunks = Chunk.objects.filter(id__in=matched_ids)
    chunk_map = {chunk.id: chunk for chunk in chunks}
    
    retrieved_texts = []
    for chunk_id in matched_ids:
        if chunk_id in chunk_map:
            retrieved_texts.append(chunk_map[chunk_id].original_text)

    joined_context = "\n\n--- NEXT CONTEXT CHUNK ---\n\n".join(retrieved_texts)

    prompt = f"""System Instruction: You are an intelligent Document Assistant. 
            Read the provided context carefully. Answer the user's question strictly using only the provided context. 
            If the answer cannot be found in the context, reply exactly with: "I cannot answer this based on the provided documents." Do not invent information.

            Context:
            {joined_context}

            User Question: {query}
            """
    return prompt, joined_context


import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def generate_answer_from_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("Please provide a valid Gemini API Key.")

    genai.configure(api_key=GEMINI_API_KEY)
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {str(e)}"
    

def calculate_file_hash(file_field) -> str:
    hasher = hashlib.sha256()
    for chunk in file_field.chunks():
        hasher.update(chunk)
    return hasher.hexdigest()


def update_document_in_index(document: Document) -> int:
    new_hash = calculate_file_hash(document.file)
    
    if document.file_hash == new_hash:
        print(f"Skipping update for '{document.title}': Content hash hasn't changed.")
        return 0

    document.file_hash = new_hash
    document.save(update_fields=['file_hash'])

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError("FAISS index not found on disk.")
        
    index = faiss.read_index(FAISS_INDEX_PATH)

    old_chunks = document.chunks.all()
    if old_chunks.exists():
        old_ids = [chunk.id for chunk in old_chunks]
        selector = faiss.IDSelectorBatch(old_ids)
        index.remove_ids(selector)
        old_chunks.delete()
        
    raw_text = extract_text_from_pdf(document)
    process_and_save_chunks(document, raw_text)
    new_chunk_count = generate_embeddings_for_document(document)

    new_chunks = document.chunks.exclude(embedding__isnull=True)
    if new_chunks.exists():
        new_ids = []
        new_vectors = []
        for chunk in new_chunks:
            new_ids.append(chunk.id)
            new_vectors.append(chunk.embedding)
            
        vectors_np = np.array(new_vectors).astype('float32')
        ids_np = np.array(new_ids).astype('int64')
        
        index.add_with_ids(vectors_np, ids_np)
        faiss.write_index(index, FAISS_INDEX_PATH)
        
    return new_chunk_count


def delete_document_from_index(document: Document) -> bool:

    if not os.path.exists(FAISS_INDEX_PATH):
        return False
        
    index = faiss.read_index(FAISS_INDEX_PATH)
    chunks = document.chunks.all()
    
    if chunks.exists():
        chunk_ids = [chunk.id for chunk in chunks]
        selector = faiss.IDSelectorBatch(chunk_ids)
        index.remove_ids(selector)
        
        faiss.write_index(index, FAISS_INDEX_PATH)

    document.delete()
    return True