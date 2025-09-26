import os
from unittest.mock import MagicMock, patch

import pytest
from config import Config
from rag_system import RAGSystem


class TestRAGSystem:
    """Integration tests for RAGSystem"""

    @pytest.fixture
    def test_rag_config(self, temp_dir):
        """Create test configuration for RAG system"""
        config = Config()
        config.CHROMA_PATH = os.path.join(temp_dir, "test_chroma")
        config.ANTHROPIC_API_KEY = "test-api-key"
        config.CHUNK_SIZE = 200
        config.CHUNK_OVERLAP = 50
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        return config

    @pytest.fixture
    def test_course_file(self, temp_dir):
        """Create a test course file"""
        course_content = """Course Title: Test Integration Course
Course Link: https://example.com/integration-test
Course Instructor: Integration Tester

Lesson 0: Introduction to Testing
Lesson Link: https://example.com/lesson0
This lesson introduces the concepts of integration testing. Integration testing is crucial for ensuring that different components work together correctly. We will explore various testing strategies and methodologies.

Lesson 1: Advanced Testing Concepts
Lesson Link: https://example.com/lesson1
In this lesson, we dive deeper into advanced testing concepts including mocking, stubbing, and test doubles. These techniques help isolate components during testing and ensure reliable test results.
"""
        test_file = os.path.join(temp_dir, "test_course.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(course_content)
        return test_file

    def test_rag_system_initialization(self, test_rag_config):
        """Test RAG system initialization"""
        rag_system = RAGSystem(test_rag_config)

        assert rag_system.config == test_rag_config
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None

        # Verify tools are registered
        tool_defs = rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_defs]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_add_course_document(self, test_rag_config, test_course_file):
        """Test adding a single course document"""
        rag_system = RAGSystem(test_rag_config)

        course, chunk_count = rag_system.add_course_document(test_course_file)

        assert course is not None
        assert course.title == "Test Integration Course"
        assert chunk_count > 0

        # Verify course was added to vector store
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 1
        assert "Test Integration Course" in analytics["course_titles"]

    def test_add_course_folder(self, test_rag_config, temp_dir):
        """Test adding courses from a folder"""
        # Create multiple test files
        for i in range(3):
            course_content = f"""Course Title: Test Course {i}
Course Link: https://example.com/course{i}
Course Instructor: Test Instructor {i}

Lesson 0: Introduction {i}
Content for course {i} lesson 0.

Lesson 1: Advanced Topics {i}
Advanced content for course {i}.
"""
            test_file = os.path.join(temp_dir, f"course{i}.txt")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(course_content)

        rag_system = RAGSystem(test_rag_config)
        courses_added, chunks_added = rag_system.add_course_folder(temp_dir)

        assert courses_added == 3
        assert chunks_added > 0

        # Verify all courses were added
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 3

    def test_add_course_folder_with_existing_courses(self, test_rag_config, temp_dir):
        """Test the problematic scenario: courses exist but chunks are missing"""
        # First, add a course
        course_content = """Course Title: Existing Course
Course Link: https://example.com/existing
Course Instructor: Existing Instructor

Lesson 0: Existing Lesson
This is existing content.
"""
        test_file = os.path.join(temp_dir, "existing.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(course_content)

        rag_system = RAGSystem(test_rag_config)

        # Add course first time
        courses_added1, chunks_added1 = rag_system.add_course_folder(temp_dir)
        assert courses_added1 == 1
        assert chunks_added1 > 0

        # Try to add same course again (simulates restart scenario)
        courses_added2, chunks_added2 = rag_system.add_course_folder(
            temp_dir, clear_existing=False
        )

        # This should skip the course since it already exists
        assert courses_added2 == 0
        assert chunks_added2 == 0

        # This is the problematic behavior! If chunks were lost somehow,
        # they won't be re-added because the course title exists in catalog.

    @patch("ai_generator.anthropic.Anthropic")
    def test_query_with_content_search(
        self, mock_anthropic_class, test_rag_config, test_course_file
    ):
        """Test querying with content that should trigger tool use"""
        # Setup mock Anthropic responses
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First response: AI wants to use search tool
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [MagicMock()]
        mock_tool_response.content[0].type = "tool_use"
        mock_tool_response.content[0].name = "search_course_content"
        mock_tool_response.content[0].id = "tool_123"
        mock_tool_response.content[0].input = {"query": "integration testing"}

        # Second response: Final answer
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = (
            "Integration testing is crucial for ensuring components work together."
        )

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Setup RAG system with test data
        rag_system = RAGSystem(test_rag_config)
        rag_system.add_course_document(test_course_file)

        # Perform query
        response, sources, source_links = rag_system.query(
            "What is integration testing?"
        )

        # Verify response
        assert (
            response
            == "Integration testing is crucial for ensuring components work together."
        )
        assert len(sources) > 0  # Should have found sources
        assert "Test Integration Course" in sources[0]

    @patch("ai_generator.anthropic.Anthropic")
    def test_query_with_empty_content(self, mock_anthropic_class, test_rag_config):
        """Test the current problematic scenario: query when no content exists"""
        # Setup mock (AI tries to search but gets no results)
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [MagicMock()]
        mock_tool_response.content[0].type = "tool_use"
        mock_tool_response.content[0].name = "search_course_content"
        mock_tool_response.content[0].id = "tool_123"
        mock_tool_response.content[0].input = {"query": "computer use"}

        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = (
            "I couldn't find any relevant content about that topic."
        )

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Setup RAG system with no content (empty vector store)
        rag_system = RAGSystem(test_rag_config)

        # Perform query (this should replicate the current "query failed" scenario)
        response, sources, source_links = rag_system.query("What is computer use?")

        # This is what happens when no content is found
        assert len(sources) == 0  # No sources found
        assert response == "I couldn't find any relevant content about that topic."

    @patch("ai_generator.anthropic.Anthropic")
    def test_query_without_tool_use(self, mock_anthropic_class, test_rag_config):
        """Test query that doesn't require tool use (general knowledge)"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Direct response without tool use
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Python is a programming language."

        mock_client.messages.create.return_value = mock_response

        rag_system = RAGSystem(test_rag_config)
        response, sources, source_links = rag_system.query("What is Python?")

        assert response == "Python is a programming language."
        assert len(sources) == 0  # No tool use, so no sources

    def test_session_management(self, test_rag_config):
        """Test session management functionality"""
        rag_system = RAGSystem(test_rag_config)

        with patch.object(
            rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.return_value = "Test response"

            # First query creates session
            response1, _, _ = rag_system.query("First query", session_id=None)
            assert response1 == "Test response"

            # Second query with same session should include history
            response2, _, _ = rag_system.query(
                "Second query", session_id="test-session"
            )

            # Verify history was passed (check if generate_response was called with history)
            calls = mock_generate.call_args_list
            if len(calls) > 1:
                # Second call should include conversation history
                assert calls[1][1].get("conversation_history") is not None

    def test_get_course_analytics(self, test_rag_config, test_course_file):
        """Test course analytics"""
        rag_system = RAGSystem(test_rag_config)

        # Initially empty
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 0
        assert len(analytics["course_titles"]) == 0

        # After adding course
        rag_system.add_course_document(test_course_file)
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 1
        assert "Test Integration Course" in analytics["course_titles"]

    def test_tool_manager_integration(self, test_rag_config, test_course_file):
        """Test that tools are properly integrated"""
        rag_system = RAGSystem(test_rag_config)
        rag_system.add_course_document(test_course_file)

        # Test search tool directly
        result = rag_system.tool_manager.execute_tool(
            "search_course_content", query="testing"
        )
        assert isinstance(result, str)
        # Should find content since we added a course
        assert len(result) > 0

        # Test outline tool directly
        result = rag_system.tool_manager.execute_tool(
            "get_course_outline", course_name="Test Integration Course"
        )
        assert isinstance(result, str)
        assert "Test Integration Course" in result
        assert "Lesson 0" in result
        assert "Lesson 1" in result

    def test_data_inconsistency_detection(self, test_rag_config, test_course_file):
        """Test detection of the real-world data inconsistency issue"""
        rag_system = RAGSystem(test_rag_config)

        # Add course document
        rag_system.add_course_document(test_course_file)

        # Verify both metadata and content exist
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 1

        # Test content search works
        result = rag_system.tool_manager.execute_tool(
            "search_course_content", query="testing"
        )
        assert "No relevant content found" not in result

        # Now simulate the problematic state: clear content but keep metadata
        # (This simulates what might have happened in the real system)
        rag_system.vector_store.course_content.delete()  # Clear content collection
        rag_system.vector_store.course_content = (
            rag_system.vector_store._create_collection("course_content")
        )

        # Course count should still be 1 (metadata exists)
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 1

        # But content search should return empty results
        result = rag_system.tool_manager.execute_tool(
            "search_course_content", query="testing"
        )
        assert "No relevant content found" in result

        # This demonstrates the exact issue in the production system!

    @patch("ai_generator.anthropic.Anthropic")
    def test_error_handling_in_query(self, mock_anthropic_class, test_rag_config):
        """Test error handling in query processing"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock API exception
        import anthropic

        mock_client.messages.create.side_effect = anthropic.APIError("API Error")

        rag_system = RAGSystem(test_rag_config)

        # Query should handle API errors gracefully
        with pytest.raises(anthropic.APIError):
            rag_system.query("Test query")

    def test_clear_existing_data(self, test_rag_config, test_course_file):
        """Test clearing existing data and reloading"""
        rag_system = RAGSystem(test_rag_config)

        # Add course
        rag_system.add_course_document(test_course_file)
        assert rag_system.get_course_analytics()["total_courses"] == 1

        # Clear and reload with clear_existing=True
        folder_path = os.path.dirname(test_course_file)
        courses_added, chunks_added = rag_system.add_course_folder(
            folder_path, clear_existing=True
        )

        # Should re-add the course
        assert courses_added == 1
        assert chunks_added > 0

        # Verify data is still there
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 1

        # And content search should work
        result = rag_system.tool_manager.execute_tool(
            "search_course_content", query="testing"
        )
        assert "No relevant content found" not in result

    def test_real_system_simulation(self, test_rag_config):
        """Test simulating the exact real system scenario"""
        # This test simulates the production system startup behavior
        rag_system = RAGSystem(test_rag_config)

        # Simulate what happens during startup with existing data
        # First, add some metadata directly (as if from a previous run)
        from models import Course, Lesson

        existing_course = Course(
            title="Existing Course",
            course_link="https://example.com/existing",
            instructor="Test Instructor",
            lessons=[
                Lesson(0, "Introduction", "Content here", "https://example.com/l0")
            ],
        )
        rag_system.vector_store.add_course_metadata(existing_course)

        # Now the system should report 1 course but 0 content chunks
        # (This mirrors the real logs: "Course already exists...skipping" + "Loaded 0 courses with 0 chunks")

        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 1  # Metadata exists

        # But content search should fail
        result = rag_system.tool_manager.execute_tool(
            "search_course_content", query="anything"
        )
        assert "No relevant content found" in result  # No content chunks

        # This is exactly the production issue!
