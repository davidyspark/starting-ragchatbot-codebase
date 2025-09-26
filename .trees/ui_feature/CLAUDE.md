# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install/sync dependencies
uv sync

# Add new dependency
uv add package-name

# Install dev dependencies (includes quality tools)
uv sync --group dev
```

### Code Quality Tools
```bash
# Format code with black and fix auto-fixable issues
./scripts/format.sh

# Run linting checks
./scripts/lint.sh

# Run tests
./scripts/test.sh

# Run full quality pipeline (format + lint + test)
./scripts/quality.sh

# Manual commands
uv run black backend/ main.py          # Format code
uv run ruff check backend/ main.py     # Check linting
uv run ruff check --fix backend/ main.py  # Fix auto-fixable issues
cd backend && uv run pytest tests/ -v  # Run tests
```

### Environment Setup
- Create `.env` file with: `ANTHROPIC_API_KEY=your_key_here`
- Uses Python 3.13+ with uv package manager

### Access Points
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) chatbot system with a clear separation between frontend and backend layers.

### Core Data Flow
1. **Document Processing**: Course documents (`docs/*.txt`) → Structured parsing → Text chunking → Vector embeddings
2. **Query Processing**: User query → Session management → Claude API with tool calling → Vector search → Response synthesis
3. **Tool-Based Search**: Claude autonomously decides when to search using `CourseSearchTool`

### Key Components

**RAG System (`backend/rag_system.py`)**
- Central orchestrator that coordinates all components
- Manages document ingestion and query processing
- Interfaces between session management, AI generation, and search tools

**Document Processing Pipeline**
- `DocumentProcessor`: Parses structured course format (Course Title/Link/Instructor + Lessons)
- Smart chunking with sentence boundaries and configurable overlap
- Context preservation: chunks include course title and lesson numbers

**Vector Store (`backend/vector_store.py`)**
- ChromaDB-based semantic search with sentence-transformers embeddings (all-MiniLM-L6-v2)
- Uses `SearchResults` dataclass for consistent result handling across search operations
- Unified search interface supporting course name and lesson number filtering
- Persistent storage in `./chroma_db/` with collections for content and metadata

**AI Integration (`backend/ai_generator.py`)**
- Claude API integration with tool calling capability
- Two-phase response: tool execution → final synthesis
- Conversation history management via session system

**Search Tools Architecture (`backend/search_tools.py`)**
- Abstract `Tool` interface for extensibility
- `CourseSearchTool` with smart course name matching and lesson filtering
- `ToolManager` handles tool registration and execution

### Configuration
- All settings centralized in `backend/config.py`
- Key parameters: chunk size (800), overlap (100), max results (5)
- ChromaDB storage: `./chroma_db/`

### Document Format
Course documents follow structured format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [lesson title]
Lesson Link: [optional url]
[lesson content]
```

### Session Management
- Conversation history maintained per session ID
- Limited history (2 exchanges) to manage context length
- Sessions created automatically on first query

## Development Notes

**Data Models (`backend/models.py`)**
- `Course`, `Lesson`, `CourseChunk` Pydantic models define the core data structure
- Course title serves as unique identifier across the system
- CourseChunk includes content, metadata, and positional indexing

**Frontend Integration**
- Vanilla JavaScript with marked.js for markdown rendering
- Static files served directly by FastAPI with no-cache headers for development
- CORS enabled for cross-origin development

**Startup Behavior**
- Documents auto-loaded from `docs/` folder on server startup
- ChromaDB collections created automatically if they don't exist
- No test framework currently implemented