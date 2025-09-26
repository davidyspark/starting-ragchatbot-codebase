import os
from unittest.mock import MagicMock, patch

import pytest
from vector_store import VectorStore


class TestVectorStore:
    """Test cases for VectorStore"""

    def test_initialization(self, temp_dir):
        """Test VectorStore initialization"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(
            chroma_path=chroma_path, embedding_model="all-MiniLM-L6-v2", max_results=5
        )

        assert store.max_results == 5
        assert store.client is not None
        assert store.course_catalog is not None
        assert store.course_content is not None

    def test_add_course_metadata(self, temp_dir, sample_course):
        """Test adding course metadata to catalog"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        # Add course metadata
        store.add_course_metadata(sample_course)

        # Verify course was added
        existing_titles = store.get_existing_course_titles()
        assert sample_course.title in existing_titles
        assert store.get_course_count() == 1

    def test_add_course_content(self, temp_dir, sample_course_chunks):
        """Test adding course content chunks"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        # Add content chunks
        store.add_course_content(sample_course_chunks)

        # Verify chunks were added by searching
        results = store.search("testing")
        assert not results.is_empty()

    def test_search_with_no_content(self, temp_dir):
        """Test search when no content has been added (current system state)"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        # Search without adding any content
        results = store.search("any query")

        # Should return empty results, not an error
        assert results.is_empty()
        assert results.error is None

    def test_search_with_content(self, temp_dir, sample_course, sample_course_chunks):
        """Test search when content exists"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        # Add both metadata and content
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Search for content
        results = store.search("testing")

        assert not results.is_empty()
        assert len(results.documents) > 0
        assert len(results.metadata) > 0

    def test_search_with_course_name_filter(
        self, temp_dir, sample_course, sample_course_chunks
    ):
        """Test search with course name filtering"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Search with exact course name
        results = store.search("testing", course_name="Test Course")
        assert not results.is_empty()

        # Search with non-existent course name
        results = store.search("testing", course_name="Nonexistent Course")
        assert results.error is not None
        assert "No course found" in results.error

    def test_search_with_lesson_filter(
        self, temp_dir, sample_course, sample_course_chunks
    ):
        """Test search with lesson number filtering"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Search with lesson number
        results = store.search("testing", lesson_number=0)
        assert not results.is_empty()

        # All returned results should be from lesson 0
        for metadata in results.metadata:
            assert metadata.get("lesson_number") == 0

    def test_resolve_course_name(self, temp_dir, sample_course):
        """Test course name resolution with partial matching"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        store.add_course_metadata(sample_course)

        # Test exact match
        resolved = store._resolve_course_name("Test Course")
        assert resolved == "Test Course"

        # Test partial match
        resolved = store._resolve_course_name("Test")
        assert resolved == "Test Course"

        # Test non-existent course
        resolved = store._resolve_course_name("Nonexistent")
        assert resolved is None

    def test_get_all_courses_metadata(self, temp_dir, sample_course):
        """Test retrieving all course metadata"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        store.add_course_metadata(sample_course)

        metadata_list = store.get_all_courses_metadata()
        assert len(metadata_list) == 1

        course_meta = metadata_list[0]
        assert course_meta["title"] == "Test Course"
        assert course_meta["instructor"] == "Test Instructor"
        assert "lessons" in course_meta
        assert len(course_meta["lessons"]) == 2

    def test_metadata_without_content_scenario(self, temp_dir, sample_course):
        """Test the current problematic scenario: metadata exists but no content"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        # Add only metadata (simulating current system state)
        store.add_course_metadata(sample_course)

        # Verify metadata exists
        assert store.get_course_count() == 1
        assert sample_course.title in store.get_existing_course_titles()

        # But search should return empty results
        results = store.search("testing frameworks")
        assert results.is_empty()
        assert results.error is None

        # This is exactly what's happening in the current system!

    def test_content_without_metadata_scenario(self, temp_dir, sample_course_chunks):
        """Test adding content without metadata"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        # Add only content chunks (no metadata)
        store.add_course_content(sample_course_chunks)

        # Search should find content
        results = store.search("testing")
        assert not results.is_empty()

        # But course count should be 0 (no metadata)
        assert store.get_course_count() == 0

    def test_clear_all_data(self, temp_dir, sample_course, sample_course_chunks):
        """Test clearing all data"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        # Add data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)

        # Verify data exists
        assert store.get_course_count() > 0
        assert not store.search("testing").is_empty()

        # Clear data
        store.clear_all_data()

        # Verify data is cleared
        assert store.get_course_count() == 0
        assert store.search("testing").is_empty()

    def test_empty_course_chunks_handling(self, temp_dir):
        """Test handling of empty chunk list"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        # Should not raise an exception
        store.add_course_content([])

        # Search should return empty results
        results = store.search("anything")
        assert results.is_empty()

    def test_get_lesson_link(self, temp_dir, sample_course):
        """Test retrieving lesson links"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        store.add_course_metadata(sample_course)

        # Get lesson link
        link = store.get_lesson_link("Test Course", 0)
        assert link == "https://example.com/lesson0"

        # Non-existent lesson
        link = store.get_lesson_link("Test Course", 999)
        assert link is None

        # Non-existent course
        link = store.get_lesson_link("Nonexistent Course", 0)
        assert link is None

    def test_get_course_link(self, temp_dir, sample_course):
        """Test retrieving course links"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        store.add_course_metadata(sample_course)

        # Get course link
        link = store.get_course_link("Test Course")
        assert link == "https://example.com/course"

        # Non-existent course
        link = store.get_course_link("Nonexistent Course")
        assert link is None

    @patch("chromadb.PersistentClient")
    def test_chromadb_connection_error(self, mock_client_class, temp_dir):
        """Test handling of ChromaDB connection errors"""
        mock_client_class.side_effect = Exception("ChromaDB connection failed")

        with pytest.raises(Exception):
            VectorStore(
                chroma_path=os.path.join(temp_dir, "test_chroma"),
                embedding_model="all-MiniLM-L6-v2",
                max_results=5,
            )

    def test_search_error_handling(self, temp_dir):
        """Test search error handling"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        # Mock the collection to raise an exception
        store.course_content.query = MagicMock(side_effect=Exception("Search failed"))

        results = store.search("test query")
        assert results.error is not None
        assert "Search error" in results.error

    def test_build_filter_logic(self, temp_dir):
        """Test the internal filter building logic"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        # No filters
        filter_dict = store._build_filter(None, None)
        assert filter_dict is None

        # Course filter only
        filter_dict = store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}

        # Lesson filter only
        filter_dict = store._build_filter(None, 1)
        assert filter_dict == {"lesson_number": 1}

        # Both filters
        filter_dict = store._build_filter("Test Course", 1)
        expected = {"$and": [{"course_title": "Test Course"}, {"lesson_number": 1}]}
        assert filter_dict == expected

    def test_real_world_data_inconsistency_detection(
        self, temp_dir, sample_course, sample_course_chunks
    ):
        """Test detection of data inconsistency that mirrors the real system"""
        chroma_path = os.path.join(temp_dir, "test_chroma")
        store = VectorStore(chroma_path, "all-MiniLM-L6-v2", 5)

        # Add metadata
        store.add_course_metadata(sample_course)

        # Simulate a partial failure where content addition fails
        # (This is likely what happened in the real system)

        # Verify inconsistent state
        assert store.get_course_count() > 0  # Metadata exists
        search_results = store.search("testing")
        assert search_results.is_empty()  # But no content

        # This test confirms the exact scenario we're seeing in production!
