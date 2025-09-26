from unittest.mock import MagicMock, patch

import pytest
from app import app, rag_system
from fastapi.testclient import TestClient


class TestAPIIntegration:
    """Test cases for API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def sample_query_request(self):
        """Sample query request data"""
        return {"query": "What is computer use?", "session_id": None}

    @pytest.fixture
    def sample_query_request_with_session(self):
        """Sample query request with session ID"""
        return {"query": "Follow-up question", "session_id": "test-session-123"}

    def test_query_endpoint_basic(self, client):
        """Test basic query endpoint functionality"""
        with patch.object(rag_system, "query") as mock_query:
            mock_query.return_value = (
                "This is a test response",
                ["Test Course"],
                ["https://example.com/course"],
            )

            with patch.object(
                rag_system.session_manager, "create_session"
            ) as mock_create:
                mock_create.return_value = "new-session-123"

                response = client.post(
                    "/api/query", json={"query": "What is testing?", "session_id": None}
                )

                assert response.status_code == 200
                data = response.json()

                assert data["answer"] == "This is a test response"
                assert data["sources"] == ["Test Course"]
                assert data["source_links"] == ["https://example.com/course"]
                assert data["session_id"] == "new-session-123"

    def test_query_endpoint_with_existing_session(self, client):
        """Test query endpoint with existing session"""
        with patch.object(rag_system, "query") as mock_query:
            mock_query.return_value = ("Follow-up response", ["Test Course"], [None])

            response = client.post(
                "/api/query",
                json={"query": "Follow-up question", "session_id": "existing-session"},
            )

            assert response.status_code == 200
            data = response.json()

            assert data["answer"] == "Follow-up response"
            assert data["session_id"] == "existing-session"

    def test_query_endpoint_with_empty_results(self, client):
        """Test query endpoint when no results are found (current issue)"""
        with patch.object(rag_system, "query") as mock_query:
            # Simulate the current failing scenario
            mock_query.return_value = (
                "I couldn't find any relevant content about that topic.",
                [],  # No sources found
                [],  # No source links
            )

            with patch.object(
                rag_system.session_manager, "create_session"
            ) as mock_create:
                mock_create.return_value = "session-123"

                response = client.post(
                    "/api/query",
                    json={"query": "What is computer use?", "session_id": None},
                )

                assert response.status_code == 200
                data = response.json()

                # This is what users see as "query failed"
                assert "couldn't find" in data["answer"]
                assert data["sources"] == []
                assert data["source_links"] == []

    def test_query_endpoint_error_handling(self, client):
        """Test query endpoint error handling"""
        with patch.object(rag_system, "query") as mock_query:
            mock_query.side_effect = Exception("Internal error")

            response = client.post(
                "/api/query", json={"query": "Test query", "session_id": None}
            )

            assert response.status_code == 500
            assert "Internal error" in response.json()["detail"]

    def test_query_endpoint_invalid_request(self, client):
        """Test query endpoint with invalid request data"""
        # Missing required query field
        response = client.post("/api/query", json={"session_id": "test-session"})

        assert response.status_code == 422  # Validation error

        # Empty query
        response = client.post("/api/query", json={"query": "", "session_id": None})

        # Should still process but might return validation error
        # The exact behavior depends on validation rules

    def test_courses_endpoint_basic(self, client):
        """Test courses endpoint basic functionality"""
        with patch.object(rag_system, "get_course_analytics") as mock_analytics:
            mock_analytics.return_value = {
                "total_courses": 2,
                "course_titles": ["Course 1", "Course 2"],
            }

            response = client.get("/api/courses")

            assert response.status_code == 200
            data = response.json()

            assert data["total_courses"] == 2
            assert data["course_titles"] == ["Course 1", "Course 2"]

    def test_courses_endpoint_empty_system(self, client):
        """Test courses endpoint with empty system (current state)"""
        with patch.object(rag_system, "get_course_analytics") as mock_analytics:
            # This simulates the current system state
            mock_analytics.return_value = {
                "total_courses": 4,  # Metadata exists
                "course_titles": [
                    "Advanced Retrieval for AI with Chroma",
                    "Prompt Compression and Query Optimization",
                    "Building Towards Computer Use with Anthropic",
                    "MCP: Build Rich-Context AI Apps with Anthropic",
                ],
            }

            response = client.get("/api/courses")

            assert response.status_code == 200
            data = response.json()

            # Courses are listed (metadata exists)
            assert data["total_courses"] == 4
            assert len(data["course_titles"]) == 4

    def test_courses_endpoint_error_handling(self, client):
        """Test courses endpoint error handling"""
        with patch.object(rag_system, "get_course_analytics") as mock_analytics:
            mock_analytics.side_effect = Exception("Database error")

            response = client.get("/api/courses")

            assert response.status_code == 500
            assert "Database error" in response.json()["detail"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_end_to_end_content_query(self, mock_anthropic_class, client):
        """Test end-to-end content query flow"""
        # Setup mock Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock tool use response
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [MagicMock()]
        mock_tool_response.content[0].type = "tool_use"
        mock_tool_response.content[0].name = "search_course_content"
        mock_tool_response.content[0].id = "tool_123"
        mock_tool_response.content[0].input = {"query": "computer use"}

        # Mock final response
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = (
            "Computer use enables automated interactions with digital interfaces."
        )

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Make request
        response = client.post(
            "/api/query", json={"query": "What is computer use?", "session_id": None}
        )

        assert response.status_code == 200
        data = response.json()

        # In the current broken system, this would likely show empty sources
        # because the search tool returns "No relevant content found"

    @patch("ai_generator.anthropic.Anthropic")
    def test_end_to_end_outline_query(self, mock_anthropic_class, client):
        """Test end-to-end course outline query"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock outline tool use
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [MagicMock()]
        mock_tool_response.content[0].type = "tool_use"
        mock_tool_response.content[0].name = "get_course_outline"
        mock_tool_response.content[0].id = "tool_123"
        mock_tool_response.content[0].input = {"course_name": "MCP"}

        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = (
            "The MCP course covers lesson 1, lesson 2, and lesson 3."
        )

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        response = client.post(
            "/api/query",
            json={
                "query": "What are the lessons in the MCP course?",
                "session_id": None,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Outline queries should work because they use metadata
        assert response.status_code == 200

    def test_cors_headers(self, client):
        """Test CORS headers are properly set"""
        response = client.options("/api/query")

        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        with patch.object(rag_system, "query") as mock_query:
            mock_query.return_value = ("Response", [], [])

            with patch.object(
                rag_system.session_manager, "create_session"
            ) as mock_create:
                mock_create.side_effect = [f"session-{i}" for i in range(5)]

                # Make multiple concurrent requests
                responses = []
                for i in range(5):
                    response = client.post(
                        "/api/query", json={"query": f"Query {i}", "session_id": None}
                    )
                    responses.append(response)

                # All should succeed
                assert all(r.status_code == 200 for r in responses)

                # Each should have different session IDs
                session_ids = [r.json()["session_id"] for r in responses]
                assert len(set(session_ids)) == 5

    def test_malformed_json_request(self, client):
        """Test handling of malformed JSON"""
        response = client.post("/api/query", data="invalid json")
        assert response.status_code == 422

    def test_large_query_request(self, client):
        """Test handling of very large queries"""
        large_query = "What is " + "testing " * 1000  # Very long query

        with patch.object(rag_system, "query") as mock_query:
            mock_query.return_value = ("Response", [], [])

            with patch.object(
                rag_system.session_manager, "create_session"
            ) as mock_create:
                mock_create.return_value = "session-123"

                response = client.post(
                    "/api/query", json={"query": large_query, "session_id": None}
                )

                # Should handle large queries (might truncate or return error)
                # The exact behavior depends on implementation

    def test_session_management_through_api(self, client):
        """Test session management through API calls"""
        with patch.object(rag_system, "query") as mock_query:
            mock_query.return_value = ("Response", [], [])

            with patch.object(
                rag_system.session_manager, "create_session"
            ) as mock_create:
                mock_create.return_value = "session-abc"

                # First request - creates session
                response1 = client.post(
                    "/api/query", json={"query": "First query", "session_id": None}
                )

                session_id = response1.json()["session_id"]

                # Second request - uses existing session
                response2 = client.post(
                    "/api/query",
                    json={"query": "Second query", "session_id": session_id},
                )

                assert response2.json()["session_id"] == session_id

    def test_static_file_serving(self, client):
        """Test that static files are served correctly"""
        # Test if frontend files are accessible
        response = client.get("/")

        # Should serve the main HTML file or return appropriate response
        # The exact behavior depends on how static files are configured
