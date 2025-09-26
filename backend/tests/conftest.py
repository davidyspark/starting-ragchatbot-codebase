import pytest
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from typing import Generator
import os
import sys

# Add backend directory to path so tests can import modules
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from vector_store import VectorStore, SearchResults
from ai_generator import AIGenerator
from document_processor import DocumentProcessor
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from models import Course, Lesson, CourseChunk
from config import Config


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test databases"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_dir: str) -> Config:
    """Create test configuration with temporary paths"""
    config = Config()
    config.CHROMA_PATH = os.path.join(temp_dir, "test_chroma_db")
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.CHUNK_SIZE = 100  # Smaller chunks for testing
    config.CHUNK_OVERLAP = 20
    config.MAX_RESULTS = 3
    return config


@pytest.fixture
def mock_vector_store() -> Mock:
    """Mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)

    # Mock search method to return empty results by default
    mock_store.search.return_value = SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )

    # Mock other methods
    mock_store._resolve_course_name.return_value = None
    mock_store.add_course_metadata.return_value = None
    mock_store.add_course_content.return_value = None
    mock_store.get_existing_course_titles.return_value = []
    mock_store.get_course_count.return_value = 0

    return mock_store


@pytest.fixture
def sample_course() -> Course:
    """Create a sample course for testing"""
    lessons = [
        Lesson(
            lesson_number=0,
            title="Introduction",
            lesson_link="https://example.com/lesson0"
        ),
        Lesson(
            lesson_number=1,
            title="Basic Concepts",
            lesson_link="https://example.com/lesson1"
        )
    ]

    return Course(
        title="Test Course",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=lessons
    )


@pytest.fixture
def sample_course_chunks(sample_course: Course) -> list[CourseChunk]:
    """Create sample course chunks for testing"""
    chunks = []

    # Create some sample content for each lesson (since Lesson model doesn't have content field)
    lesson_contents = {
        0: "Welcome to this course about testing. This lesson introduces basic concepts of testing frameworks and methodologies.",
        1: "This lesson covers basic concepts of testing frameworks. We explore unit testing, integration testing, and best practices."
    }

    for lesson in sample_course.lessons:
        lesson_content = lesson_contents.get(lesson.lesson_number, f"Content for lesson {lesson.lesson_number}")

        # Split lesson content into smaller chunks for testing
        words = lesson_content.split()
        chunk_size = 5  # 5 words per chunk for testing

        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)

            chunks.append(CourseChunk(
                course_title=sample_course.title,
                lesson_number=lesson.lesson_number,
                content=chunk_content,
                chunk_index=len(chunks)
            ))

    return chunks


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for AI generator testing"""
    mock_client = MagicMock()

    # Mock a basic text response
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "This is a test response."
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def sample_search_results() -> SearchResults:
    """Create sample search results for testing"""
    return SearchResults(
        documents=[
            "This is a sample document about testing.",
            "Another document with testing information."
        ],
        metadata=[
            {
                "course_title": "Test Course",
                "lesson_number": 0,
                "chunk_index": 0
            },
            {
                "course_title": "Test Course",
                "lesson_number": 1,
                "chunk_index": 1
            }
        ],
        distances=[0.1, 0.15]
    )


@pytest.fixture
def empty_search_results() -> SearchResults:
    """Create empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results() -> SearchResults:
    """Create error search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Test search error"
    )


@pytest.fixture
def document_processor(test_config: Config) -> DocumentProcessor:
    """Create document processor for testing"""
    return DocumentProcessor(test_config.CHUNK_SIZE, test_config.CHUNK_OVERLAP)


@pytest.fixture
def course_search_tool(mock_vector_store: Mock) -> CourseSearchTool:
    """Create CourseSearchTool with mock vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_outline_tool(mock_vector_store: Mock) -> CourseOutlineTool:
    """Create CourseOutlineTool with mock vector store"""
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager() -> ToolManager:
    """Create ToolManager for testing"""
    return ToolManager()


@pytest.fixture
def sample_course_document() -> str:
    """Sample course document content for testing"""
    return """Course Title: Test Course
Course Link: https://example.com/test-course
Course Instructor: Test Instructor

Lesson 0: Introduction
Lesson Link: https://example.com/lesson0
Welcome to this test course. This lesson introduces basic concepts.
We will cover fundamental principles and practical applications.

Lesson 1: Advanced Topics
Lesson Link: https://example.com/lesson1
This lesson covers more advanced topics in the subject.
You will learn about complex implementations and best practices.
"""


# API Testing Fixtures

@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    mock_rag = Mock()

    # Mock query method
    mock_rag.query.return_value = (
        "This is a test response from the RAG system.",
        ["Test Course"],
        ["https://example.com/course"]
    )

    # Mock session manager
    mock_session_manager = Mock()
    mock_session_manager.create_session.return_value = "test-session-123"
    mock_rag.session_manager = mock_session_manager

    # Mock course analytics
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }

    return mock_rag


@pytest.fixture
def test_client(mock_rag_system):
    """Create FastAPI test client with mocked dependencies"""
    from fastapi.testclient import TestClient

    # Create a test app without static file mounting to avoid frontend dependency
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Create test app
    test_app = FastAPI(title="Test RAG System")

    # Add CORS middleware
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        source_links: List[Optional[str]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # API endpoints with mocked RAG system
    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources, source_links = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                source_links=source_links,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/")
    async def root():
        return {"message": "Test RAG API"}

    return TestClient(test_app)


@pytest.fixture
def sample_query_requests():
    """Sample API request payloads for testing"""
    return {
        "basic_query": {
            "query": "What is machine learning?",
            "session_id": None
        },
        "with_session": {
            "query": "Tell me more about that topic",
            "session_id": "existing-session-123"
        },
        "empty_query": {
            "query": "",
            "session_id": None
        },
        "long_query": {
            "query": "What is " + "machine learning " * 100,
            "session_id": None
        }
    }


@pytest.fixture
def mock_api_responses():
    """Mock API response data for testing"""
    return {
        "successful_query": {
            "answer": "Machine learning is a subset of artificial intelligence.",
            "sources": ["AI Course", "ML Basics"],
            "source_links": ["https://example.com/ai", "https://example.com/ml"],
            "session_id": "session-123"
        },
        "no_results": {
            "answer": "I couldn't find any relevant information about that topic.",
            "sources": [],
            "source_links": [],
            "session_id": "session-123"
        },
        "course_stats": {
            "total_courses": 3,
            "course_titles": ["AI Course", "ML Basics", "Deep Learning"]
        }
    }