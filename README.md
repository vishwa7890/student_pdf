# PDF RAG System 🤖📄

A comprehensive PDF processing and Retrieval-Augmented Generation (RAG) system that enables intelligent querying of PDF documents using advanced NLP techniques and LLaMA 3.2.

## 🌟 Features

### Core Capabilities
- **Multi-method PDF Text Extraction**: Uses pdfplumber (primary), PyPDF2 (backup), and EasyOCR (for scanned PDFs)
- **Advanced Semantic Chunking**: Intelligent text segmentation that preserves context
- **Vector Embeddings**: Uses BAAI/bge-large-en-v1.5 sentence transformer for high-quality embeddings
- **FAISS Vector Store**: Efficient similarity search with separate indexes per chat session
- **RAG-powered Querying**: Retrieves relevant chunks and generates contextual answers using LLaMA 3.2
- **Multi-PDF Support**: Handle multiple documents per chat session
- **Web Document Summarization**: Generate summaries from web URLs
- **Modern Web Interface**: Beautiful, responsive UI for easy interaction

### Technical Highlights
- **Modular Architecture**: Each PDF gets its own FAISS index for scalability
- **OCR Fallback**: Handles scanned PDFs with EasyOCR integration
- **Context-Aware Responses**: Answers strictly based on retrieved document context
- **Persistent Storage**: Indexes and metadata saved to disk for session continuity
- **RESTful API**: Complete FastAPI backend with comprehensive endpoints

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (for model loading)

### Installation

1. **Clone or download the project files**
   ```bash
   cd pdf_2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**
   ```bash
   python app.py
   ```

4. **Open the web interface**
   - Navigate to `http://localhost:8000` in your browser
   - Or open the `index.html` file directly

## 📖 Usage Guide

### 1. Upload and Process PDFs
- Enter a unique chat session ID (e.g., "session_001")
- Upload your PDF file via drag-and-drop or file picker
- The system will extract text, create semantic chunks, and build embeddings
- Wait for the success confirmation

### 2. Chat with Your PDFs
- Type questions about your uploaded document
- The system retrieves relevant chunks and generates contextual answers
- Responses are formatted in markdown for better readability
- All answers are based strictly on the document content

### 3. Web Document Summarization
- Enter any web URL in the summarization section
- Choose between "Executive Summary" or "Detailed Summary"
- Get AI-generated summaries of web content

## 🔧 API Endpoints

### Core Endpoints

#### `POST /upload/`
Upload and process PDF files
```json
{
  "file": "PDF file",
  "chat_id": "unique_session_id"
}
```

#### `POST /chat`
Query uploaded PDFs using RAG
```json
{
  "message": "What is the main topic of the document?",
  "chat_id": "unique_session_id"
}
```

#### `POST /websurf/generate-summary/`
Generate summaries from web URLs
```json
{
  "url": "https://example.com/article",
  "summary_type": "executive"
}
```

#### `GET /` and `GET /test-connection`
Health check endpoints

## 🏗️ Architecture

### Data Flow
1. **PDF Upload** → Text extraction → Semantic chunking → Embedding generation → FAISS index creation
2. **Query Processing** → Query embedding → FAISS similarity search → Context retrieval → LLaMA response generation
3. **Response Delivery** → Formatted markdown response with source attribution

### Key Components

#### Text Processing Pipeline
- **pdfplumber**: Primary text extraction with layout preservation
- **PyPDF2**: Backup extraction method
- **EasyOCR**: OCR for scanned documents with PyMuPDF image conversion
- **Semantic Chunking**: Context-aware text segmentation with overlap

#### Vector Processing
- **Sentence Transformer**: BAAI/bge-large-en-v1.5 for embeddings
- **FAISS**: IndexFlatIP with L2 normalization for cosine similarity
- **Persistent Storage**: Binary index files and pickle metadata

#### LLM Integration
- **OpenRouter API**: Access to LLaMA 3.2-3B-Instruct
- **Contextual Prompting**: Retrieval-augmented generation with strict context adherence
- **Response Formatting**: Markdown output with structured presentation

## 📁 File Structure

```
pdf_2/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── index.html         # Web interface
├── README.md          # This documentation
├── faiss_index_*.bin  # FAISS indexes (generated)
└── metadata_*.pickle  # Document metadata (generated)
```

## ⚙️ Configuration

### Environment Variables
The system uses the following configuration:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OPENROUTER_MODEL`: LLaMA model identifier
- `OPENROUTER_URL`: API endpoint URL

### Model Configuration
- **Embedding Model**: BAAI/bge-large-en-v1.5 (768 dimensions)
- **LLM Model**: meta-llama/llama-3.2-3b-instruct:free
- **Chunk Size**: 1000 characters with 200 character overlap
- **Retrieval**: Top-5 relevant chunks per query

## 🔍 Advanced Features

### Multi-Document Querying
- Upload multiple PDFs to the same chat session
- System combines and searches across all documents
- Responses indicate source document for transparency

### Semantic Chunking Algorithm
- Paragraph-aware splitting with sentence boundary preservation
- Overlap handling for context continuity
- Minimum chunk size filtering (50+ characters)
- Handles edge cases like oversized paragraphs

### OCR Processing
- Automatic fallback for scanned PDFs
- PyMuPDF for PDF-to-image conversion
- EasyOCR for text recognition
- Temporary file management with cleanup

## 🚨 Troubleshooting

### Common Issues

**Model Loading Errors**
- Ensure sufficient RAM (4GB+)
- Check internet connection for model downloads
- Verify PyTorch installation

**PDF Processing Failures**
- Try different PDF files to isolate issues
- Check file permissions and corruption
- Ensure PDF contains extractable text

**API Connection Issues**
- Verify OpenRouter API key validity
- Check network connectivity
- Confirm model availability

**Performance Issues**
- Reduce chunk size for faster processing
- Limit concurrent uploads
- Monitor system resources

### Debug Mode
Enable detailed logging by modifying the logging level in `app.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Metrics

### Typical Processing Times
- **Text PDF (10 pages)**: 5-15 seconds
- **Scanned PDF (10 pages)**: 30-60 seconds (OCR)
- **Query Response**: 2-5 seconds
- **Web Summary**: 10-20 seconds

### Resource Usage
- **RAM**: 2-4GB (models loaded)
- **Storage**: ~50MB per processed PDF
- **CPU**: Moderate during processing, low during idle

## 🤝 Contributing

This system is designed to be modular and extensible. Key areas for enhancement:
- Additional PDF extraction methods
- Alternative embedding models
- Enhanced chunking strategies
- Multi-language support
- Advanced summarization techniques

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with all API terms of service and model licenses.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation
3. Verify system requirements
4. Test with sample documents

---

**Built with ❤️ using FastAPI, FAISS, Sentence Transformers, and LLaMA 3.2**
