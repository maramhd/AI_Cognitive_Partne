# AI Cognitive Partner V2

A NotebookLM-inspired academic AI assistant that answers questions exclusively from your uploaded sources.

## Features

- **Multi-format support**: PDF, TXT, MD, DOCX, PPTX, XLSX, images (PNG/JPG/WEBP), video (MP4/MOV/AVI)
- **Source-grounded answers**: Every answer is backed by citations from your documents
- **Three-panel UI**: Sources (left), Chat (center), Studio (right)
- **Local-first**: No paid API keys required. Uses local LLM (Ollama) and local embeddings (multilingual-e5-small)
- **Persistent**: Conversations and sources persist across restarts
- **Precise citations**: Page numbers, slide numbers, sheet names, timestamps
- **Embedding compatibility guard**: RAG stops when the active query model does not match the model recorded for the selected course; it does not rebuild or re-embed the index automatically

## Quick Start

```cmd
cd "H:\AI Cognitive Partner\ai-engineering-hub-main\ai-cognitive-partner-v2"
.venv\Scripts\activate
python -m streamlit run app.py
```

## Enable LLM (optional but recommended)

1. Install Ollama from https://ollama.com
2. Run: `ollama pull qwen3:1.7b`
3. Restart the application

## System Requirements

- Python 3.11+
- 8 GB RAM (minimum)
- Windows 10+

## Storage

All data is stored on H: drive under the project directory.

## Dependencies

See `pyproject.toml` for the full dependency list.

## Known Limitations

- OCR requires Tesseract OCR to be installed (not currently installed)
- Video processing requires FFmpeg to be installed (not currently installed)
- Video transcription uses faster-whisper (available)
- Studio features (summary, flashcards, quiz) are planned but not yet implemented
- LLM generation requires Ollama with a pulled model
- Image visual analysis beyond OCR is not yet implemented