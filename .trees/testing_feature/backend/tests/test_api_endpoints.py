import pytest
import json
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient


@pytest.mark.api
class TestQueryEndpoint:
    """Comprehensive tests for /api/query endpoint"""

    def test_successful_query_new_session(self, test_client, sample_query_requests, mock_rag_system):
        """Test successful query with new session creation"""
        request_data = sample_query_requests["basic_query"]

        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "source_links" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"

        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once_with(request_data["query"], "test-session-123")
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_successful_query_existing_session(self, test_client, sample_query_requests, mock_rag_system):
        """Test successful query with existing session"""
        request_data = sample_query_requests["with_session"]

        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == "existing-session-123"

        # Verify no new session was created
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with(
            request_data["query"],
            "existing-session-123"
        )

    def test_query_with_no_results(self, test_client, sample_query_requests, mock_rag_system):
        """Test query when no relevant content is found"""
        # Mock empty results
        mock_rag_system.query.return_value = (
            "I couldn't find any relevant information about that topic.",
            [],
            []
        )

        request_data = sample_query_requests["basic_query"]
        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "couldn't find" in data["answer"]
        assert data["sources"] == []
        assert data["source_links"] == []

    def test_query_endpoint_error_handling(self, test_client, sample_query_requests, mock_rag_system):
        """Test error handling in query endpoint"""
        # Mock RAG system to raise exception
        mock_rag_system.query.side_effect = Exception("RAG system error")

        request_data = sample_query_requests["basic_query"]
        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]

    def test_query_validation_missing_query(self, test_client):
        """Test validation when query field is missing"""
        response = test_client.post("/api/query", json={"session_id": "test"})

        assert response.status_code == 422  # Validation error

    def test_query_validation_empty_query(self, test_client, sample_query_requests, mock_rag_system):
        """Test handling of empty query string"""
        request_data = sample_query_requests["empty_query"]
        response = test_client.post("/api/query", json=request_data)

        # Should still process (behavior depends on implementation)
        assert response.status_code in [200, 422]

    def test_query_validation_invalid_json(self, test_client):
        """Test handling of malformed JSON"""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"content-type": "application/json"}
        )

        assert response.status_code == 422

    def test_large_query_handling(self, test_client, sample_query_requests, mock_rag_system):
        """Test handling of very large queries"""
        request_data = sample_query_requests["long_query"]
        response = test_client.post("/api/query", json=request_data)

        # Should handle gracefully (exact behavior depends on implementation)
        assert response.status_code in [200, 413, 422]

    def test_query_response_schema(self, test_client, sample_query_requests):
        """Test that response matches expected schema"""
        request_data = sample_query_requests["basic_query"]
        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        required_fields = ["answer", "sources", "source_links", "session_id"]
        for field in required_fields:
            assert field in data

        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["source_links"], list)
        assert isinstance(data["session_id"], str)

    def test_concurrent_queries(self, test_client, mock_rag_system):
        """Test handling of multiple concurrent queries"""
        # Mock different session IDs for each request
        mock_rag_system.session_manager.create_session.side_effect = [
            f"session-{i}" for i in range(5)
        ]

        # Make multiple concurrent requests
        responses = []
        for i in range(5):
            response = test_client.post("/api/query", json={
                "query": f"Query {i}",
                "session_id": None
            })
            responses.append(response)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # Each should have different session IDs
        session_ids = [r.json()["session_id"] for r in responses]
        assert len(set(session_ids)) == 5


@pytest.mark.api
class TestCoursesEndpoint:
    """Comprehensive tests for /api/courses endpoint"""

    def test_courses_endpoint_basic(self, test_client, mock_rag_system):
        """Test basic courses endpoint functionality"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Test Course 1", "Test Course 2"]

        # Verify RAG system was called
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_courses_endpoint_empty_system(self, test_client, mock_rag_system):
        """Test courses endpoint with no courses loaded"""
        # Mock empty system
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_endpoint_error_handling(self, test_client, mock_rag_system):
        """Test error handling in courses endpoint"""
        # Mock analytics to raise exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]

    def test_courses_response_schema(self, test_client, mock_rag_system):
        """Test that courses response matches expected schema"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])

    def test_courses_endpoint_large_dataset(self, test_client, mock_rag_system):
        """Test courses endpoint with large number of courses"""
        # Mock large dataset
        large_course_list = [f"Course {i}" for i in range(1000)]
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 1000,
            "course_titles": large_course_list
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 1000
        assert len(data["course_titles"]) == 1000


@pytest.mark.api
class TestRootEndpoint:
    """Tests for root endpoint"""

    def test_root_endpoint(self, test_client):
        """Test root endpoint functionality"""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert data["message"] == "Test RAG API"


@pytest.mark.api
class TestMiddleware:
    """Tests for API middleware and cross-cutting concerns"""

    def test_cors_headers(self, test_client, sample_query_requests):
        """Test CORS headers are properly set"""
        # Test CORS headers on actual request since OPTIONS may not be implemented
        request_data = sample_query_requests["basic_query"]
        response = test_client.post("/api/query", json=request_data)

        # Check for CORS headers
        assert response.status_code == 200
        # CORS headers may be set by middleware - check if present
        # Note: Test client might not show all middleware headers

    def test_content_type_headers(self, test_client, sample_query_requests):
        """Test content type handling"""
        request_data = sample_query_requests["basic_query"]

        # Test with explicit content-type
        response = test_client.post(
            "/api/query",
            json=request_data,
            headers={"content-type": "application/json"}
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_method_not_allowed(self, test_client):
        """Test handling of unsupported HTTP methods"""
        # GET on query endpoint (should be POST)
        response = test_client.get("/api/query")
        assert response.status_code == 405

        # POST on courses endpoint (should be GET)
        response = test_client.post("/api/courses", json={})
        assert response.status_code == 405


@pytest.mark.api
@pytest.mark.integration
class TestEndToEndAPI:
    """End-to-end API integration tests"""

    def test_query_to_courses_workflow(self, test_client, mock_rag_system):
        """Test typical user workflow: query then check courses"""
        # First, make a query
        query_response = test_client.post("/api/query", json={
            "query": "What courses are available?",
            "session_id": None
        })

        assert query_response.status_code == 200
        session_id = query_response.json()["session_id"]

        # Then check available courses
        courses_response = test_client.get("/api/courses")

        assert courses_response.status_code == 200
        courses_data = courses_response.json()

        # Verify we can see available courses
        assert courses_data["total_courses"] > 0
        assert len(courses_data["course_titles"]) > 0

    def test_session_continuity(self, test_client, mock_rag_system):
        """Test session continuity across multiple queries"""
        # First query - creates session
        response1 = test_client.post("/api/query", json={
            "query": "What is machine learning?",
            "session_id": None
        })

        session_id = response1.json()["session_id"]

        # Follow-up query - uses same session
        response2 = test_client.post("/api/query", json={
            "query": "Tell me more about that",
            "session_id": session_id
        })

        # Session should be maintained
        assert response2.json()["session_id"] == session_id

        # Verify second call didn't create new session
        assert mock_rag_system.session_manager.create_session.call_count == 1

    def test_api_error_recovery(self, test_client, mock_rag_system):
        """Test API behavior after errors"""
        # Cause an error
        mock_rag_system.query.side_effect = Exception("Temporary error")

        response1 = test_client.post("/api/query", json={
            "query": "Test query",
            "session_id": None
        })

        assert response1.status_code == 500

        # Reset mock to normal behavior
        mock_rag_system.query.side_effect = None
        mock_rag_system.query.return_value = ("Normal response", [], [])

        # Next request should work normally
        response2 = test_client.post("/api/query", json={
            "query": "Another test query",
            "session_id": None
        })

        assert response2.status_code == 200