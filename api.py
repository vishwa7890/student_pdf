import asyncio
try:
    import aiohttp
except ImportError:
    aiohttp = None
import os
import pickle
import tempfile
import time
import random
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pdfplumber
import PyPDF2
import easyocr
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import requests
import re
import json
from datetime import datetime
from dotenv import load_dotenv

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    # Set manual seed for reproducibility
    torch.cuda.manual_seed_all(42)
else:
    logger.info("No GPU available, using CPU")

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup and clean up on shutdown"""
    logger.info("Starting PDF RAG System...")
    initialize_models()
    logger.info("PDF RAG System started successfully!")
    yield  # This is where the application runs
    # Any cleanup code can go here

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="PDF RAG System",
    description="Advanced PDF processing and RAG querying system",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory=".")

# Add CORS middleware with comprehensive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Type"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Load environment variables from .env file (handled in config section below as well)
load_dotenv()

# API key check is performed in the generic provider config section below

# Directory for storing uploaded files and FAISS indexes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_indexes")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# Log the absolute paths for verification
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Upload directory: {UPLOAD_DIR}")
logger.info(f"FAISS index directory: {FAISS_INDEX_DIR}")

# Verify upload directory is writable
if not os.access(UPLOAD_DIR, os.W_OK):
    logger.error(f"Upload directory is not writable: {UPLOAD_DIR}")
    raise PermissionError(f"Cannot write to upload directory: {UPLOAD_DIR}")

# Global variables
sentence_transformer = None
ocr_reader = None
chat_document_mapping = {}
faiss_indexes = {}
document_metadata = {}
chat_model_selection: Dict[str, str] = {}

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    chat_id: str
    model: Optional[str] = None

class SummaryRequest(BaseModel):
    url: str
    summary_type: str = "executive"
    model: Optional[str] = None

def initialize_models():
    """Initialize the sentence transformer and OCR reader with device support"""
    global sentence_transformer, ocr_reader
    
    try:
        logger.info(f"Initializing models on {device}...")
        
        # Initialize sentence transformer with device
        logger.info("Loading sentence transformer model...")
        sentence_transformer = SentenceTransformer(
            'BAAI/bge-large-en-v1.5',
            device=str(device)  # Convert device to string for sentence-transformers
        )
        sentence_transformer = sentence_transformer.to(device)
        logger.info(f"Sentence transformer loaded successfully on {device}")
        
        # Initialize OCR reader with device configuration
        logger.info("Loading OCR reader...")
        ocr_reader = easyocr.Reader(
            ['en'],
            gpu=torch.cuda.is_available(),  # Automatically use GPU if available
            download_enabled=True
        )
        logger.info(f"OCR reader loaded successfully with {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        # Test the models
        test_embedding = sentence_transformer.encode("Test sentence", convert_to_tensor=True)
        logger.info(f"Test embedding shape: {test_embedding.shape}")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}", exc_info=True)
        raise

def clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/]', '', text)
    
    # Remove multiple consecutive punctuation
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    
    # Strip and return
    return text.strip()

def improved_semantic_text_chunking(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Advanced semantic text chunking that preserves context
    """
    if not text:
        return []
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap by taking last few sentences
                sentences = current_chunk.split('.')
                overlap_text = '. '.join(sentences[-2:]) if len(sentences) > 2 else current_chunk[-overlap:]
                current_chunk = overlap_text + " " + paragraph
            else:
                # Paragraph itself is too long, split by sentences
                sentences = paragraph.split('.')
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) > chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                        else:
                            # Even single sentence is too long, force split
                            chunks.append(sentence[:chunk_size])
                            temp_chunk = sentence[chunk_size:]
                    else:
                        temp_chunk += sentence + "."
                
                current_chunk = temp_chunk
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter out very short chunks

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF using multiple methods with fallbacks
    """
    text = ""
    
    # Method 1: pdfplumber (primary)
    try:
        logger.info("Attempting text extraction with pdfplumber...")
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if text.strip():
            logger.info(f"Successfully extracted {len(text)} characters with pdfplumber")
            return clean_text(text)
            
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
    
    # Method 2: PyPDF2 (backup)
    try:
        logger.info("Attempting text extraction with PyPDF2...")
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if text.strip():
            logger.info(f"Successfully extracted {len(text)} characters with PyPDF2")
            return clean_text(text)
            
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}")
    
    # Method 3: OCR with EasyOCR (for scanned PDFs)
    try:
        logger.info("Attempting OCR extraction with EasyOCR...")
        import fitz  # PyMuPDF for converting PDF to images
        
        pdf_document = fitz.open(file_path)
        ocr_text = ""
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            
            # Save temporary image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
                temp_img.write(img_data)
                temp_img_path = temp_img.name
            
            try:
                # Perform OCR
                results = ocr_reader.readtext(temp_img_path)
                page_text = " ".join([result[1] for result in results])
                ocr_text += page_text + "\n"
            finally:
                os.unlink(temp_img_path)
        
        pdf_document.close()
        
        if ocr_text.strip():
            logger.info(f"Successfully extracted {len(ocr_text)} characters with OCR")
            return clean_text(ocr_text)
            
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
    
    raise HTTPException(status_code=400, detail="Failed to extract text from PDF using all available methods")

def create_embeddings(texts: List[str]) -> np.ndarray:
    """Create embeddings for text chunks with device support"""
    try:
        if not texts:
            return np.array([])
            
        # Convert texts to embeddings on the specified device
        with torch.no_grad():
            embeddings = sentence_transformer.encode(
                texts, 
                convert_to_tensor=True,
                device=device,
                show_progress_bar=True
            )
            # Convert to numpy array and ensure it's float32 for FAISS
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy().astype('float32')
                
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
            
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}", exc_info=True)
        raise

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create FAISS index from embeddings"""
    if embeddings.size == 0:
        raise ValueError("No embeddings provided")
        
    # Ensure embeddings are float32 and contiguous
    if not isinstance(embeddings, np.ndarray) or embeddings.dtype != np.float32:
        embeddings = np.array(embeddings, dtype=np.float32)
    
    # Ensure embeddings are 2D
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Create a copy to avoid modifying the input
        embeddings_norm = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_norm)
        
        # Add to index
        index.add(embeddings_norm)
        
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        logger.error(f"Embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}")
        logger.error(f"Embeddings type: {type(embeddings)}")
        if hasattr(embeddings, 'dtype'):
            logger.error(f"Embeddings dtype: {embeddings.dtype}")
        raise

def save_index_and_metadata(chat_id: str, index, chunks: List[Dict], filename: str):
    """Save FAISS index and metadata to disk in the FAISS_INDEX_DIR"""
    try:
        # Ensure the directory exists
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(FAISS_INDEX_DIR, f"{chat_id}_index.faiss")
        faiss.write_index(index, index_path)
        
        # Save metadata
        metadata = {
            'chunks': chunks,
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(FAISS_INDEX_DIR, f"{chat_id}_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving index and metadata: {str(e)}")
        return False

def load_index_and_metadata(chat_id: str):
    """Load FAISS index and metadata from disk"""
    try:
        # Construct paths
        index_path = os.path.join(FAISS_INDEX_DIR, f"{chat_id}_index.faiss")
        metadata_path = os.path.join(FAISS_INDEX_DIR, f"{chat_id}_metadata.pkl")
        
        # Check if files exist
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning(f"Index or metadata file not found for chat_id: {chat_id}")
            return None, None
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        logger.info(f"Loaded index from {index_path} and metadata from {metadata_path}")
        return index, metadata
    except Exception as e:
        logger.error(f"Error loading index and metadata: {str(e)}", exc_info=True)
        return None, None

def delete_index_and_metadata(chat_id: str):
    """Delete FAISS index and metadata files from the FAISS_INDEX_DIR"""
    try:
        index_path = os.path.join(FAISS_INDEX_DIR, f"{chat_id}_index.faiss")
        metadata_path = os.path.join(FAISS_INDEX_DIR, f"{chat_id}_metadata.pkl")
        
        deleted = False
        
        if os.path.exists(index_path):
            os.remove(index_path)
            logger.info(f"Deleted index: {index_path}")
            deleted = True
            
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            logger.info(f"Deleted metadata: {metadata_path}")
            deleted = True
            
        if not deleted:
            logger.warning(f"No index or metadata found for chat_id: {chat_id}")
            
        return deleted
    except Exception as e:
        logger.error(f"Error deleting index and metadata: {str(e)}")
        return False

def retrieve_relevant_chunks(query: str, chat_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks using FAISS similarity search"""
    try:
        # Load index and metadata
        index, metadata = load_index_and_metadata(chat_id)
        
        if index is None or metadata is None:
            return []
        
        # Create query embedding
        query_embedding = sentence_transformer.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        scores, indices = index.search(query_embedding, min(top_k, len(metadata['chunks'])))
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(metadata['chunks']):
                results.append({
                    'chunk': metadata['chunks'][idx],
                    'score': float(score),
                    'rank': i + 1,
                    'filename': metadata['filename']
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {e}")
        return []

# Load environment variables
load_dotenv()

# Provider configuration (generic, no vendor names).
# Primary provider configuration
PRIMARY_API_KEY = os.getenv('PRIMARY_API_KEY')
if not PRIMARY_API_KEY:
    raise ValueError("PRIMARY_API_KEY environment variable not set")

PRIMARY_API_URL = os.getenv('PRIMARY_API_URL')
PRIMARY_MODEL = os.getenv('PRIMARY_MODEL')

# Maximum number of retries for API calls
MAX_RETRIES = 3

# Alternate provider configuration (optional)
ALT_API_KEY = os.getenv('ALT_API_KEY')
ALT_API_URL = os.getenv('ALT_API_URL')
# Default fallback mapping when UI sends a generic alias like '<ALT_PREFIX>/alias'
ALT_DEFAULT_MODEL = os.getenv('ALT_DEFAULT_MODEL')
# Routing prefix for alternate provider, e.g., 'openai' or any custom identifier (not hardcoded)
ALT_PREFIX = os.getenv('ALT_PREFIX')

def _map_alt_model(model: Optional[str]) -> Optional[str]:
    """Map a UI-provided model string with a configurable prefix to an alternate provider model id.

    Example: if ALT_PREFIX='alt', 'alt/some-model' -> 'some-model'.
    If remainder matches an alias convention, optionally map to ALT_DEFAULT_MODEL.
    """
    if not model:
        return None
    model = model.strip()
    if ALT_PREFIX and model.startswith(f"{ALT_PREFIX}/"):
        name = model.split('/', 1)[1]
        # Allow aliasing through ALT_DEFAULT_MODEL if a generic alias is used
        if ALT_DEFAULT_MODEL and name.lower() in {"default", "alias", "turbo"}:
            return ALT_DEFAULT_MODEL
        # Otherwise assume the remainder is a valid alternate provider model id
        return name
    return None

async def query_alt_llm(prompt: str, retry_count: int = 0, alt_model: Optional[str] = None) -> str:
    """
    Query the alternate provider's chat completions API.
    """
    if not ALT_API_KEY:
        logger.error("Upstream API key is not configured")
        return "Error: Upstream API key is not configured"
    if retry_count >= MAX_RETRIES:
        logger.error(f"Maximum retry attempts ({MAX_RETRIES}) exceeded")
        return "Error: Maximum number of retry attempts reached"

    try:
        if retry_count > 0:
            delay = min(2.0 * (2 ** (retry_count - 1)), 30.0)
            await asyncio.sleep(delay)

        model_to_use = ((alt_model or ALT_DEFAULT_MODEL) or "").strip()
        if not model_to_use:
            logger.error("No model configured for upstream provider")
            return "Error: No model configured for upstream provider"
        logger.info(f"Sending request to upstream provider with model: {model_to_use}")

        headers = {
            "Authorization": f"Bearer {ALT_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
        }

        if aiohttp is not None:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ALT_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=300
                ) as response:
                    response_data = await response.json()

                    if response.status != 200:
                        error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                        logger.error(f"Upstream API error: {error_msg}")
                        if response.status == 429:
                            retry_after = int(response.headers.get('Retry-After', 5))
                            logger.info(f"Rate limited. Retrying after {retry_after} seconds...")
                            await asyncio.sleep(retry_after)
                            return await query_alt_llm(prompt, retry_count + 1, alt_model=model_to_use)
                        return f"Error: {error_msg}"

                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        return response_data['choices'][0]['message']['content'].strip()
                    else:
                        logger.error(f"Unexpected upstream response format: {response_data}")
                        return "Error: Unexpected response format from API"
        else:
            def _do_request():
                r = requests.post(
                    ALT_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                try:
                    data = r.json()
                except Exception:
                    data = {"error": {"message": f"Non-JSON response: {r.text[:200]}"}}
                return r.status_code, data, r.headers

            status, response_data, resp_headers = await asyncio.to_thread(_do_request)
            if status != 200:
                error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                logger.error(f"Upstream API error: {error_msg}")
                if status == 429:
                    retry_after = int(resp_headers.get('Retry-After', 5))
                    logger.info(f"Rate limited. Retrying after {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    return await query_alt_llm(prompt, retry_count + 1, alt_model=model_to_use)
                return f"Error: {error_msg}"

            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content'].strip()
            else:
                logger.error(f"Unexpected upstream response format: {response_data}")
                return "Error: Unexpected response format from API"
    except asyncio.TimeoutError:
        logger.warning(f"Upstream request timed out. Retry attempt {retry_count + 1}/{MAX_RETRIES}")
        return await query_alt_llm(prompt, retry_count + 1, alt_model=alt_model)
    except Exception as e:
        logger.error(f"Upstream HTTP request error: {str(e)}")
        if retry_count < MAX_RETRIES - 1:
            return await query_alt_llm(prompt, retry_count + 1, alt_model=alt_model)
        return f"Error: {str(e)}"

async def query_llm(prompt: str, retry_count: int = 0, model: Optional[str] = None) -> str:
    """
    Query the OpenRouter API for text generation.
    
    Args:
        prompt: The input prompt to send to the model
        retry_count: Number of retry attempts made so far
        
    Returns:
        str: The generated text response from the model
    """
    if not PRIMARY_API_KEY:
        logger.error("Primary provider API key is not configured")
        return "Error: Primary provider API key is not configured"
        
    if retry_count >= MAX_RETRIES:
        logger.error(f"Maximum retry attempts ({MAX_RETRIES}) exceeded")
        return "Error: Maximum number of retry attempts reached"
        
    try:
        # Add a small delay between requests
        if retry_count > 0:
            delay = min(2.0 * (2 ** (retry_count - 1)), 30.0)  # Exponential backoff, max 30s
            await asyncio.sleep(delay)

        # Provider routing: if model indicates an alternate provider via ALT_PREFIX, route accordingly
        routed_alt_model = _map_alt_model(model)
        if routed_alt_model:
            logger.info(f"Routing to alternate provider with model: {routed_alt_model}")
            return await query_alt_llm(prompt, retry_count=retry_count, alt_model=routed_alt_model)

        selected_model = ((model or PRIMARY_MODEL) or "").strip()
        if not selected_model:
            logger.error("No model configured for primary provider")
            return "Error: No model configured for primary provider"
        logger.info(f"Sending request to primary provider API with model: {selected_model} (override={'yes' if model else 'no'})")

        headers = {
            "Authorization": f"Bearer {PRIMARY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "stop": ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"]
        }
        
        if aiohttp is not None:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    PRIMARY_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=300  # 5 minute timeout
                ) as response:
                    response_data = await response.json()
                    
                    if response.status != 200:
                        error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                        logger.error(f"API error: {error_msg}")
                        
                        # Handle rate limiting
                        if response.status == 429:
                            retry_after = int(response.headers.get('Retry-After', 5))
                            logger.info(f"Rate limited. Retrying after {retry_after} seconds...")
                            await asyncio.sleep(retry_after)
                            return await query_llm(prompt, retry_count + 1, model=selected_model)
                            
                        return f"Error: {error_msg}"
                    
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        return response_data['choices'][0]['message']['content'].strip()
                    else:
                        logger.error(f"Unexpected response format: {response_data}")
                        return "Error: Unexpected response format from API"
        else:
            # Fallback to requests in a background thread if aiohttp is not available
            def _do_request():
                r = requests.post(
                    PRIMARY_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                try:
                    data = r.json()
                except Exception:
                    data = {"error": {"message": f"Non-JSON response: {r.text[:200]}"}}
                return r.status_code, data, r.headers

            status, response_data, resp_headers = await asyncio.to_thread(_do_request)

            if status != 200:
                error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                logger.error(f"OpenRouter API error: {error_msg}")
                if status == 429:
                    retry_after = int(resp_headers.get('Retry-After', 5))
                    logger.info(f"Rate limited. Retrying after {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    return await query_llm(prompt, retry_count + 1)
                return f"Error: {error_msg}"

            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content'].strip()
            else:
                logger.error(f"Unexpected response format: {response_data}")
                return "Error: Unexpected response format from API"
                    
    except asyncio.TimeoutError:
        logger.warning(f"Request timed out. Retry attempt {retry_count + 1}/{MAX_RETRIES}")
        return await query_llm(prompt, retry_count + 1)
        
    except Exception as e:
        logger.error(f"HTTP request error: {str(e)}")
        if retry_count < MAX_RETRIES - 1:
            return await query_llm(prompt, retry_count + 1, model=model)
        return "I'm having trouble processing your request. Please try again later."

async def summarize_with_rag(query: str, chat_id: str, model: Optional[str] = None) -> str:
    """Generate accurate and well-structured answers using RAG with OpenRouter
    
    Args:
        query: The user's question or query
        chat_id: Unique identifier for the chat session
        
    Returns:
        str: Formatted, accurate response with proper citations and structure
    """
    try:
        # Enhanced subject categories with educational standards alignment
        subject_categories = {
            'math': {
                'keywords': ['calculate', 'solve', 'find', 'prove', 'equation', 'formula', 'theorem',
                           '+', '-', '*', '/', '=', '^', '√', 'square root', 'pythagorean', 'vector',
                           'algebra', 'geometry', 'calculus', 'trigonometry', 'derivative', 'integral',
                           'polynomial', 'matrix', 'probability', 'statistics', 'proof'],
                'grade_levels': {
                    'elementary': ['addition', 'subtraction', 'multiplication', 'division', 'fractions'],
                    'middle_school': ['pre-algebra', 'ratios', 'percentages', 'basic geometry'],
                    'high_school': ['algebra', 'geometry', 'trigonometry', 'pre-calculus'],
                    'college': ['calculus', 'linear algebra', 'differential equations', 'abstract algebra']
                },
                'common_standards': ['CCSS.MATH', 'NCTM', 'TEKS', 'AERO']
            },
            'science': {
                'keywords': ['experiment', 'hypothesis', 'conclusion', 'analyze', 'observe', 'predict',
                           'physics', 'chemistry', 'biology', 'ecology', 'data', 'graph', 'table',
                           'lab', 'scientific method', 'theory', 'law', 'principle', 'cell', 'atom',
                           'energy', 'force', 'reaction', 'organism', 'evolution', 'genetics'],
                'subjects': {
                    'physics': ['mechanics', 'thermodynamics', 'electromagnetism', 'quantum', 'relativity'],
                    'chemistry': ['organic', 'inorganic', 'physical', 'analytical', 'biochemistry'],
                    'biology': ['cell biology', 'genetics', 'ecology', 'microbiology', 'zoology', 'botany']
                },
                'safety': ['wear goggles', 'lab safety', 'chemical handling', 'emergency procedures']
            },
            'humanities': {
                'keywords': ['explain', 'describe', 'compare', 'contrast', 'discuss', 'evaluate',
                           'history', 'literature', 'philosophy', 'religion', 'art', 'music',
                           'analysis', 'interpret', 'critique', 'context', 'theme', 'character'],
                'eras': {
                    'ancient': ['mesopotamia', 'egypt', 'greece', 'rome', 'indus valley'],
                    'medieval': ['middle ages', 'byzantine', 'islamic golden age', 'feudalism'],
                    'modern': ['renaissance', 'enlightenment', 'industrial revolution', 'contemporary']
                },
                'critical_thinking': ['bias', 'perspective', 'primary source', 'secondary source']
            },
            'languages': {
                'keywords': ['translate', 'grammar', 'vocabulary', 'sentence', 'paragraph', 'essay',
                           'french', 'spanish', 'german', 'english', 'language', 'conjugate',
                           'pronoun', 'verb', 'noun', 'adjective', 'syntax', 'phonetics'],
                'proficiency_levels': ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'],
                'skills': ['listening', 'speaking', 'reading', 'writing'],
                'certifications': ['TOEFL', 'IELTS', 'DELE', 'DELF', 'TestDaF']
            },
            'social_sciences': {
                'keywords': ['society', 'culture', 'politics', 'economy', 'psychology', 'sociology',
                           'geography', 'anthropology', 'government', 'economics', 'behavior',
                           'research', 'statistics', 'survey', 'case study', 'theory'],
                'methods': ['qualitative', 'quantitative', 'mixed methods', 'ethnography', 'survey'],
                'ethics': ['informed consent', 'confidentiality', 'bias', 'objectivity']
            }
        }
        
        # Enhanced subject detection with confidence scoring
        query_lower = query.lower()
        subject_scores = {subject: 0 for subject in subject_categories}
        
        # Score each subject based on keyword matches
        for subject, data in subject_categories.items():
            # Base keywords
            for keyword in data['keywords']:
                if keyword in query_lower:
                    subject_scores[subject] += 1
            
            # Check for subject-specific terms
            if 'subjects' in data:
                for sub, terms in data['subjects'].items():
                    if any(term in query_lower for term in terms):
                        subject_scores[subject] += 2  # Higher weight for specific terms
        
        # Get subjects with highest scores
        max_score = max(subject_scores.values())
        detected_subjects = [subj for subj, score in subject_scores.items() 
                           if score > 0 and score >= max_score * 0.7]  # 70% threshold
        
        # Set query type flags with confidence levels
        is_math_query = 'math' in detected_subjects
        is_science_query = 'science' in detected_subjects
        is_humanities_query = 'humanities' in detected_subjects
        is_language_query = 'languages' in detected_subjects
        
        # Knowledge Graph based educational level and concept detection
        class KnowledgeGraph:
            def __init__(self):
                self.concepts = {}
                self.relationships = {}
                self.educational_levels = {
                    'elementary': {'grades': 'K-5', 'age_range': '5-10', 'complexity': 'basic'},
                    'middle_school': {'grades': '6-8', 'age_range': '11-13', 'complexity': 'intermediate'},
                    'high_school': {'grades': '9-12', 'age_range': '14-18', 'complexity': 'advanced'},
                    'college': {'grades': '13+', 'age_range': '18+', 'complexity': 'expert'}
                }
                
                # Initialize with core educational concepts
                self._initialize_kg()
            
            def _initialize_kg(self):
                # Math concepts with prerequisites and difficulty levels
                math_concepts = {
                    'addition': {'level': 'elementary', 'prerequisites': [], 'related': ['arithmetic']},
                    'algebra': {'level': 'middle_school', 'prerequisites': ['arithmetic'], 'related': ['equations']},
                    'calculus': {'level': 'high_school', 'prerequisites': ['algebra', 'trigonometry'], 'related': ['derivatives', 'integrals']},
                    'linear_algebra': {'level': 'college', 'prerequisites': ['algebra'], 'related': ['matrices', 'vector_spaces']}
                }
                
                # Add concepts to KG
                for concept, data in math_concepts.items():
                    self.add_concept(concept, 'math', data)
                
                # Add relationships
                self.add_relationship('addition', 'prerequisite_for', 'algebra')
                self.add_relationship('algebra', 'prerequisite_for', 'calculus')
                self.add_relationship('algebra', 'prerequisite_for', 'linear_algebra')
            
            def add_concept(self, concept, domain, metadata):
                self.concepts[concept] = {'domain': domain, **metadata}
                if concept not in self.relationships:
                    self.relationships[concept] = {}
            
            def add_relationship(self, source, relation, target):
                if source not in self.relationships:
                    self.relationships[source] = {}
                if relation not in self.relationships[source]:
                    self.relationships[source][relation] = []
                self.relationships[source][relation].append(target)
            
            def get_concept_level(self, concept):
                return self.concepts.get(concept, {}).get('level')
            
            def get_prerequisites(self, concept):
                return self.relationships.get(concept, {}).get('prerequisite_for', [])
            
            def find_related_concepts(self, query):
                related = []
                for concept, data in self.concepts.items():
                    if concept in query.lower():
                        related.append({
                            'concept': concept,
                            'level': data.get('level'),
                            'domain': data.get('domain'),
                            'related': data.get('related', [])
                        })
                return related
        
        # Initialize Knowledge Graph
        kg = KnowledgeGraph()
        
        # Detect educational level and concepts
        educational_level = None
        detected_concepts = kg.find_related_concepts(query_lower)
        
        # Determine educational level based on concepts
        if detected_concepts:
            concept_levels = [concept['level'] for concept in detected_concepts if concept.get('level')]
            if concept_levels:
                # Use the highest level concept found
                level_priority = ['college', 'high_school', 'middle_school', 'elementary']
                for level in level_priority:
                    if level in concept_levels:
                        educational_level = level
                        break
        
        # Fallback to keyword-based level detection if no concepts found
        if not educational_level:
            level_indicators = {
                'elementary': ['elementary', 'grade school', 'primary school', 'grades k-5'],
                'middle_school': ['middle school', 'junior high', 'grades 6-8'],
                'high_school': ['high school', 'secondary school', 'grades 9-12'],
                'college': ['college', 'university', 'undergraduate', 'graduate']
            }
            
            for level, terms in level_indicators.items():
                if any(term in query_lower for term in terms):
                    educational_level = level
                    break
        
        # Function to parse vector from text
        def parse_vector(text):
            try:
                # Look for patterns like (x, y) or [x, y]
                import re
                match = re.search(r'[\[\(]\s*(-?\d+\s*,\s*-?\d+)\s*[\]\)]', text)
                if match:
                    x, y = map(int, match.group(1).split(','))
                    return (x, y)
                return None
            except:
                return None
        
        # Retrieve relevant chunks with increased top_k for better context
        relevant_chunks = retrieve_relevant_chunks(query, chat_id, top_k=7 if not is_math_query else 10)
        
        if not relevant_chunks and not is_math_query:
            return (
                "## I couldn't find enough relevant information to answer your question.\n\n"
                "### Suggestions:\n"
                "- Try rephrasing your question\n"
                "- Be more specific with your query\n"
                "- Check if the document contains the information you're looking for"
            )
        
        # Create structured context with clear citations
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            source = f"{chunk.get('filename', 'Document')}, Page {chunk.get('page', 'N/A')}"
            context_parts.append(
                f"### Source [{i}]: {source}\n"
                f"```\n{chunk['chunk'].strip()}\n```"
            )
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt with specialized instructions for math problems
        if is_math_query:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert mathematics tutor and vector analyst. Provide clear, step-by-step solutions to mathematical problems.

## Instructions for Math Problems:
1. **Restate the question** clearly at the beginning.
2. **Identify** the relevant mathematical concepts and formulas.
3. **Show all steps** of your calculations clearly.
4. **Box the final answer** using markdown: `\boxed{{answer}}`
5. **Explain each step** in simple terms.
6. **Use proper mathematical notation** with LaTeX when needed.
7. **Check your work** for accuracy.

## Context (if available):
{context}

## Question:
{query}

## Your Response:
1. **Restate the question**: """

            # Add the rest of the prompt with specific instructions based on query type
            if is_science_query:
                prompt += f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert science tutor analyzing an ecology experiment about mangroves and saltwort plants. 

## Guidelines:
1. **Base your response** strictly on the provided experimental data and context.
2. **For data questions**:
   - Reference specific figures, tables, or data points
   - Explain trends and patterns in the data
   - Discuss statistical significance if mentioned
3. **For experimental design**:
   - Identify independent and dependent variables
   - Note sample sizes and controls
   - Consider potential confounding factors
4. **For conclusions**:
   - Support claims with evidence from the data
   - Acknowledge limitations of the study
   - Suggest possible follow-up experiments

## Context:
{context}

## Question:
{query}

## Your Response:
1. **Understanding the Question**: Clarify what's being asked
2. **Relevant Data**: Identify and present key data points
3. **Analysis**: Explain what the data shows
4. **Conclusion**: Answer the question based on evidence
5. **Limitations**: Note any data gaps or uncertainties

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                return
                
            # Subject-specific prompts
            if is_math_query:
                subject_guidance = """You are an expert mathematics tutor. Provide clear, step-by-step solutions that help students understand mathematical concepts thoroughly.

## Guidelines for Math Problems:
1. **Show all work**: Display every step of calculations
2. **Use proper notation**: 
   - Vectors: <x, y> or (x, y)
   - Equations: Use proper mathematical symbols
   - Units: Include and convert when necessary
3. **Explain concepts**: Briefly explain the why behind each step
4. **Verify solutions**: Check answers for reasonableness
5. **Use visual aids**: Suggest diagrams when helpful (e.g., for geometry)"""
            elif is_science_query:
                subject_guidance = """You are a science educator. Provide accurate, evidence-based explanations that help students understand scientific concepts.

## Guidelines for Science Problems:
1. **State the concept**: Clearly identify the scientific principle
2. **Show calculations**: Include formulas and units
3. **Explain reasoning**: Connect concepts to real-world applications
4. **Use proper terminology**: Define technical terms
5. **Include diagrams**: Suggest visual representations when helpful"""
            elif is_humanities_query:
                subject_guidance = """You are a humanities expert. Provide thoughtful, well-structured analysis of texts and concepts.

## Guidelines for Humanities:
1. **Contextualize**: Provide historical/cultural context
2. **Analyze**: Break down themes and arguments
3. **Cite evidence**: Reference specific examples
4. **Compare perspectives**: Show different viewpoints
5. **Use proper citations**: When referencing sources"""
            else:
                subject_guidance = """You are an expert educational assistant. Provide clear, accurate, and helpful explanations across all academic subjects.

## General Guidelines:
1. **Be precise**: Use accurate terminology
2. **Be clear**: Explain concepts simply
3. **Be thorough**: Cover all aspects of the question
4. **Be educational**: Help the student learn, not just get answers"""

            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{subject_guidance}

## For All Responses:
- Use markdown formatting for clarity
- Include examples when helpful
- Suggest additional resources for further learning
- Check facts before presenting information

## Question:
{query}

## Your Response:
1. **Understanding the Problem**:
   - Restate the question in your own words
   - Identify what's being asked

2. **Key Concepts**:
   - Relevant vector principles
   - Any formulas needed

3. **Step-by-Step Solution**:
   - Show each calculation
   - Explain each step clearly

4. **Verification**:
   - Check if the answer makes sense
   - Alternative approaches if applicable

5. **Final Answer**:
   - Boxed and clearly labeled
   - With proper units if applicable

6. **Additional Insights**:
   - Real-world applications
   - Common mistakes to avoid
   - Related concepts
"""
            
            # Check for vector addition problems
            if any(term in query.lower() for term in ['vector', 'component', 'resultant']):
                # Enhanced vector extraction with better pattern matching
                vectors = {}
                vector_patterns = [
                    (r'vector\s+[a-zA-Z]\s*=\s*[<\(\[]\s*(-?\d+)\s*,\s*(-?\d+)\s*[>\)\]]', 1),  # vector A = <3,4>
                    (r'[a-zA-Z]\s*=\s*[<\(\[]\s*(-?\d+)\s*,\s*(-?\d+)\s*[>\)\]]', 0),  # A = (3,4)
                    (r'vector\s+([a-zA-Z])\b.*?[<\(\[]\s*(-?\d+)\s*,\s*(-?\d+)\s*[>\)\]]', 1),  # vector A with components after
                ]
                
                for pattern, group_offset in vector_patterns:
                    for match in re.finditer(pattern, query, re.IGNORECASE):
                        var_name = match.group(1).strip() if group_offset else match.group(0)[0]
                        x, y = map(int, match.groups()[-2:])
                        vectors[var_name.upper()] = (x, y)
                
                # If no vectors found with patterns, try the original method
                if not vectors:
                    vector_terms = ['a =', 'b =', 'c =', 'vector a', 'vector b']
                    for term in vector_terms:
                        if term in query:
                            vec = parse_vector(query[query.find(term):])
                            if vec:
                                vectors[term[0].upper()] = vec
                
                # Add specific vector information if found
                vector_info = ''
                if vectors:
                    vector_info = '\n### Vectors Identified:\n' + '\n'.join([f'- {k} = {v}' for k, v in vectors.items()])
                
                prompt += f"""
3. **Vector Addition Rules**:
   - To add vectors, add their corresponding components
   - For vectors a = (a₁, a₂) and b = (b₁, b₂):
     a + b = (a₁ + b₁, a₂ + b₂)
   - For subtraction: a - b = (a₁ - b₁, a₂ - b₂)
   - Pay close attention to negative signs in components

4. **Solution**:{vector_info}
   - Step 1: Verify the given vectors from the problem
   - Step 2: Write down their components with correct signs
   - Step 3: Add/subtract corresponding components
   - Step 4: Present the final vector

5. **Verification**:
   - Double-check each component's sign
   - Verify the direction makes sense
   - Consider drawing a diagram
   - Check if the result matches expectations

6. **Common Mistakes to Avoid**:
   - Incorrect sign handling (especially with negative numbers)
   - Mixing up x and y components
   - Adding when subtraction is needed or vice versa
   - Misreading vector components from the problem

7. **Final Answer**:
   \boxed{{(x, y)}}
"""
            else:
                prompt += """
3. **Solution**:
   - Step 1: [Explain first step]
     - Calculation: [Show calculation]
   - Step 2: [Explain next step]
     - Calculation: [Show calculation]
   - [Continue as needed]

4. **Verification**:
   - [Optional: Verify the answer]
   - [Check for reasonableness]

5. **Final Answer**:
   \boxed{{answer}}

6. **Additional Notes**:
   - [Any additional context or alternative methods]
   - [Common mistakes to avoid]
"""
        else:
            # Worksheet-optimized prompt per user's template (enhanced citations + enrichment)
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a tutor helping a student complete a worksheet based on the document "iCivics the Enlightenment". Use the provided context to:

1. Identify any worksheet-style questions (short answer, multiple choice, fill-in-the-blank, crossword).
2. Answer them using only the information from the context.

Format your answers clearly, labeling each one appropriately (Q1/A1, Q2/A2, ...). Use simple markdown. If wording in the context does not match verbatim, infer the answer from closely related information in the context and cite it. If information is genuinely missing, say so briefly and cite the nearest relevant snippet.

Citation policy:
- Always include inline citations grounded in the provided context.
- Prefer explicit labels when available, e.g., [Source: The Social Contract], [Source: Vocabulary], [Source: Page 2], rather than only numeric [5]. If explicit labels are not present, keep numeric citations.

Polish if present in context (optional but encouraged):
- Include the Enlightenment timeframe (e.g., 1715–1789) if it appears in the context.
- When answers reference ideas like natural rights, social contract, or separation of powers, mention the associated thinker (e.g., Locke, Rousseau, Montesquieu) if the context contains it.

Enrichment (bonus, keep brief and only if supported by context):
- Add a short "Why it matters today" bullet under each answer.
- If key thinkers are discussed, include a small table summarizing thinker and main idea.

Context Source: PDF of the iCivics worksheet.
Focus only on answering the questions found in the document.

## Context
{context}

## Task
{query}

## Your Response
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        # Get response from OpenRouter
        response = await query_llm(prompt, model=model)
        
        # Post-process the response for better formatting
        response = response.strip()
        
        # Special handling for math responses
        if is_math_query:
            # Ensure the response has proper mathematical formatting
            response = response.replace('\n', '\n    ').replace('    \n', '\n')  # Fix indentation
            
            # Ensure the final answer is boxed if not already
            if '\boxed' not in response and 'Final Answer' in response:
                response = response.replace('Final Answer:', 'Final Answer: \\boxed{')
                if '\n' in response[response.find('\\boxed{'):]:
                    response = response.replace('\n', '}', 1)
                else:
                    response += '}'
        
        # General response formatting
        if response:
            # Ensure response ends with appropriate punctuation
            if not response.endswith(('.', '!', '?')):
                response = response.rstrip('.,!?;:') + '.'
                
            # Add double newlines before headers for better markdown rendering
            response = response.replace('\n#', '\n\n#')
            
            # Ensure sources section is properly formatted if present
            if '## Sources' in response:
                response = response.replace('## Sources', '\n## Sources')
        
        return response or "I couldn't generate a response. Please try rephrasing your question."
        
    except Exception as e:
        logger.error(f"Error in RAG summarization: {e}", exc_info=True)
        return (
            "## I encountered an error while processing your request.\n\n"
            "### Please try again later or rephrase your question.\n"
            "If the issue persists, contact support with the following details:\n"
            f"- Error: {str(e)[:200]}"
        )

# API Routes

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main UI"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return {"status": "error", "message": "UI files not found. Please ensure index.html exists in the root directory."}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "PDF RAG System is running"}

@app.get("/test-connection")
async def test_connection():
    """Test API connection"""
    return {"status": "connected", "timestamp": datetime.now().isoformat()}

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...), chat_id: str = Form(None)):
    """
    Handle PDF file uploads.
    Saves the file to the uploads directory and processes it for text extraction.
    """
    # Log the upload attempt
    logger.info(f"Received upload request for file: {file.filename}")
    logger.info(f"Content type: {file.content_type}")
    logger.info(f"Chat ID: {chat_id if chat_id else 'Not provided'}")
    # Generate a random chat ID if none provided
    if not chat_id or chat_id == 'undefined':
        import random
        import string
        chat_id = 'chat_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        logger.info(f"Generated new chat ID: {chat_id}")
    
    logger.info(f"Upload request received. File: {file.filename}, Chat ID: {chat_id}")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are allowed"
        )
    
    # Create a safe filename
    safe_filename = "".join(c if c.isalnum() or c in '._- ' else '_' for c in file.filename)
    
    # Ensure the filename has a .pdf extension
    if not safe_filename.lower().endswith('.pdf'):
        safe_filename += '.pdf'
    
    # Create a unique filename if the file already exists
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    counter = 1
    base, ext = os.path.splitext(safe_filename)
    
    while os.path.exists(file_path):
        safe_filename = f"{base}_{counter}{ext}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        counter += 1
    
    logger.info(f"Saving file to: {file_path}")
    logger.info(f"Safe filename: {safe_filename}")
    logger.info(f"Full path: {os.path.abspath(file_path)}")
    
    try:
        logger.info(f"Saving uploaded file to: {file_path}")
        
        # Save the file in chunks to handle large files
        total_bytes = 0
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                buffer.write(chunk)
                total_bytes += len(chunk)
        
        logger.info(f"Successfully saved {total_bytes} bytes to {file_path}")
        
        # Verify the file was saved correctly
        if not os.path.exists(file_path):
            raise IOError(f"Failed to save file to {file_path}")
            
        file_size = os.path.getsize(file_path)
        logger.info(f"File saved successfully. Size: {file_size} bytes")
        
        logger.info(f"File saved successfully. Size: {os.path.getsize(file_path)} bytes")
        
        # Verify the file is not empty
        if os.path.getsize(file_path) == 0:
            # Keep the empty file for debugging
            logger.warning(f"Empty file uploaded: {file_path}")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Extract text from PDF
        logger.info(f"Extracting text from: {file_path}")
        text = extract_text_from_pdf(file_path)
        if not text or not text.strip():
            logger.error("No text could be extracted from the PDF")
            raise HTTPException(
                status_code=400, 
                detail="No text could be extracted from the PDF. The file might be scanned or corrupted."
            )
        
        logger.info(f"Extracted {len(text)} characters of text")
        
        # Create semantic chunks
        logger.info("Creating semantic chunks...")
        chunks = improved_semantic_text_chunking(text)
        
        if not chunks:
            logger.error("Failed to create text chunks")
            raise HTTPException(
                status_code=400, 
                detail="Failed to process document content. The file might be corrupted or in an unsupported format."
            )
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        
        # Create embeddings
        logger.info("Generating embeddings...")
        try:
            embeddings = create_embeddings(chunks)
            logger.info(f"Generated {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing document: {str(e)}"
            )
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        try:
            index = create_faiss_index(embeddings)
            logger.info("FAISS index created successfully")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error creating document index. Please try again."
            )
        
        try:
            # Save index and metadata
            logger.info("Saving index and metadata...")
            # Save the file information
            save_index_and_metadata(chat_id, index, chunks, safe_filename)
                
            # Update chat document mapping
            if chat_id not in chat_document_mapping:
                chat_document_mapping[chat_id] = []
            
            # Add file info to chat document mapping
            file_info = {
                'filename': file.filename,
                'upload_time': datetime.now().isoformat(),
                'chunks_created': len(chunks),
                'text_length': len(text)
            }
            chat_document_mapping[chat_id].append(file_info)
                
            logger.info(f"PDF processing complete. Chat ID: {chat_id}, Chunks: {len(chunks)}")
                
            # Return success response with chat_id and file info
            return JSONResponse({
                    "status": "success",
                    "message": f"PDF '{file.filename}' processed successfully",
                    "chat_id": chat_id,
                    "filename": safe_filename,
                    "chunks_created": len(chunks),
                    "text_length": len(text)
                })
                
        except Exception as e:
            logger.error(f"Error saving index/metadata: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error saving document data. Please try again."
            )
            
    except HTTPException as http_exc:
        logger.error(f"HTTP Exception during file upload: {str(http_exc)}")
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {str(e)}", exc_info=True)
        detail = str(e) if str(e) else "An unknown error occurred"
        logger.info(f"Keeping file for debugging: {file_path}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {detail}")

@app.get("/get_pdf/{filename}")
async def get_pdf(filename: str):
    """Serve a PDF file for viewing"""
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        return FileResponse(file_path, media_type='application/pdf')
        
    except Exception as e:
        logger.error(f"Error serving PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving PDF: {str(e)}")

@app.delete("/delete_pdf/{filename}")
async def delete_pdf(filename: str):
    """Delete an uploaded PDF file and its associated index"""
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        # Delete the file
        os.remove(file_path)
        
        # Find and delete the associated index
        # The chat_id is the filename without extension
        chat_id = os.path.splitext(filename)[0]
        delete_index_and_metadata(chat_id)
        
        return {"status": "success", "message": f"Successfully deleted {filename}"}
        
    except Exception as e:
        logger.error(f"Error deleting PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")

@app.post("/chat/")
async def chat_with_pdf(request: ChatRequest):
    """Handle chat messages with context from uploaded PDFs"""
    try:
        logger.info(f"Received chat request: {request.message}")
        logger.info(f"Model override received: {request.model}")
        # Track and log model switching per chat_id
        selected_model = (request.model or os.getenv('PRIMARY_MODEL', '')).strip()
        prev_model = chat_model_selection.get(request.chat_id)
        if prev_model and selected_model and prev_model != selected_model:
            logger.info(
                f"Model switched for chat_id={request.chat_id}: {prev_model} -> {selected_model}"
            )
        elif not prev_model and selected_model:
            logger.info(
                f"Model set for chat_id={request.chat_id}: {selected_model}"
            )
        if selected_model:
            chat_model_selection[request.chat_id] = selected_model
        
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
            
        # Generate response using RAG (with optional per-request model override)
        response = await summarize_with_rag(request.message, request.chat_id, model=request.model)
        
        return {
            "response": response,
            "chat_id": request.chat_id,
            "model_used": (request.model or os.getenv('PRIMARY_MODEL')),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException as he:
        logger.error(f"HTTP error in chat: {str(he)}")
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat message: {str(e)}"
        )

@app.post("/api/summarize_web")
async def generate_web_summary(request: SummaryRequest):
    """Generate summary for web documents using OpenRouter"""
    try:
        # Fetch web content
        response = requests.get(request.url, timeout=30)
        response.raise_for_status()
        
        # Extract text content (basic implementation)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        text = clean_text(text)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found at the provided URL")
        
        # Create summary prompt based on type
        if request.summary_type == "executive":
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant that creates executive summaries of web content.

Please provide a concise executive summary of the following web content. 
Focus on key points, main arguments, and important conclusions.

Content:
{text[:4000]}

Provide a clear and concise executive summary in markdown format.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        else:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant that creates detailed summaries of web content.

Please provide a detailed section-by-section summary of the following web content.
Break down the content into logical sections and summarize each section.

Content:
{text[:4000]}

Provide a structured summary with clear sections in markdown format.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        # Generate summary using OpenRouter (with optional per-request model)
        summary = await query_llm(prompt, model=request.model)
        
        # Post-process the summary
        summary = summary.strip()
        if not any(summary.endswith(p) for p in ('.', '!', '?')):
            summary += '.'
            
        return {
            "status": "success",
            "summary": summary,
            "url": request.url,
            "summary_type": request.summary_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except requests.RequestException as e:
        logger.error(f"Error fetching web content: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=f"Error fetching web content: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error generating web summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating summary: {str(e)}"
        )

# The startup event is now handled by the lifespan context manager above

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8030)
