import pytest
from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""

    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is properly structured"""
        definition = course_search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_execute_with_successful_search(
        self, mock_vector_store, sample_search_results
    ):
        """Test execute method with successful search results"""
        # Setup mock to return sample results
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=None
        )

        # Check result format
        assert isinstance(result, str)
        assert "Test Course" in result
        assert "This is a sample document" in result

    def test_execute_with_course_name_filter(
        self, mock_vector_store, sample_search_results
    ):
        """Test execute method with course name filter"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Test Course")

        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="Test Course", lesson_number=None
        )

    def test_execute_with_lesson_number_filter(
        self, mock_vector_store, sample_search_results
    ):
        """Test execute method with lesson number filter"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test execute method with both course name and lesson number"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Test Course", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="Test Course", lesson_number=1
        )

    def test_execute_with_empty_results(self, mock_vector_store, empty_search_results):
        """Test execute method with empty search results"""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        assert result == "No relevant content found."

    def test_execute_with_empty_results_and_filters(
        self, mock_vector_store, empty_search_results
    ):
        """Test execute method with empty results and filters applied"""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Test Course", lesson_number=1)

        expected = "No relevant content found in course 'Test Course' in lesson 1."
        assert result == expected

    def test_execute_with_search_error(self, mock_vector_store, error_search_results):
        """Test execute method when search returns an error"""
        mock_vector_store.search.return_value = error_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        assert result == "Test search error"

    def test_format_results_with_lesson_numbers(self, mock_vector_store):
        """Test that results are properly formatted with lesson context"""
        # Create search results with lesson numbers
        results = SearchResults(
            documents=["Content from lesson 1", "Content from lesson 2"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )

        mock_vector_store.search.return_value = results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        assert "[Test Course - Lesson 1]" in result
        assert "[Test Course - Lesson 2]" in result
        assert "Content from lesson 1" in result
        assert "Content from lesson 2" in result

    def test_format_results_without_lesson_numbers(self, mock_vector_store):
        """Test formatting when lesson numbers are missing"""
        results = SearchResults(
            documents=["General course content"],
            metadata=[{"course_title": "Test Course"}],  # No lesson_number
            distances=[0.1],
        )

        mock_vector_store.search.return_value = results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        assert "[Test Course]" in result  # No lesson number in header
        assert "General course content" in result

    def test_source_tracking(self, mock_vector_store, sample_search_results):
        """Test that sources are properly tracked for UI display"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute("test query")

        # Check that sources were stored
        assert len(tool.last_sources) == 2
        assert "Test Course - Lesson 0" in tool.last_sources
        assert "Test Course - Lesson 1" in tool.last_sources

        # Check that source links were stored
        assert hasattr(tool, "last_source_links")
        assert len(tool.last_source_links) == 2

    def test_current_system_with_empty_content(self, mock_vector_store):
        """Test the current problematic scenario: metadata exists but no content"""
        # Simulate the current system state: empty content results
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("What is computer use?")

        # This should return the "No relevant content found" message
        # which explains why users see "query failed"
        assert result == "No relevant content found."

    def test_exception_handling_in_vector_store(self, mock_vector_store):
        """Test handling when vector store search throws an exception"""
        mock_vector_store.search.side_effect = Exception(
            "Vector store connection failed"
        )

        tool = CourseSearchTool(mock_vector_store)

        # This should not raise an exception but return an error message
        # The actual behavior depends on whether the vector store catches exceptions
        with pytest.raises(Exception):
            tool.execute("test query")

    def test_malformed_metadata_handling(self, mock_vector_store):
        """Test handling of malformed metadata in search results"""
        results = SearchResults(
            documents=["Test content"],
            metadata=[{"invalid": "metadata"}],  # Missing required fields
            distances=[0.1],
        )

        mock_vector_store.search.return_value = results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        # Should handle missing course_title gracefully
        assert "[unknown]" in result
        assert "Test content" in result
