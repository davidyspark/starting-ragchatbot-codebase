from unittest.mock import MagicMock, Mock, patch

import pytest
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator"""

    def test_initialization(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")

        assert generator.model == "claude-sonnet-4-20250514"
        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test basic response generation without tools"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "This is a test response."
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response("Test query")

        assert result == "This is a test response."
        mock_client.messages.create.assert_called_once()

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test response generation with conversation history"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Response with history."
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Follow-up query", conversation_history="Previous conversation context"
        )

        assert result == "Response with history."

        # Verify that conversation history was included in system prompt
        call_args = mock_client.messages.create.call_args[1]
        assert "Previous conversation context" in call_args["system"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_class):
        """Test response generation with tools available but not used"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Direct answer without tools."
        mock_response.stop_reason = "end_turn"  # No tool use
        mock_client.messages.create.return_value = mock_response

        # Create mock tools
        mock_tools = [{"name": "test_tool", "description": "A test tool"}]
        mock_tool_manager = Mock()

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "General knowledge query", tools=mock_tools, tool_manager=mock_tool_manager
        )

        assert result == "Direct answer without tools."

        # Verify tools were provided in API call
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == mock_tools

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_tool_use(self, mock_anthropic_class):
        """Test response generation when AI decides to use a tool"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First response: AI wants to use a tool
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [MagicMock()]
        mock_tool_response.content[0].type = "tool_use"
        mock_tool_response.content[0].name = "search_course_content"
        mock_tool_response.content[0].id = "tool_call_123"
        mock_tool_response.content[0].input = {"query": "What is computer use?"}

        # Second response: Final answer after tool execution
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "Based on the search, computer use is..."

        # Setup mock to return different responses on different calls
        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result about computer use"

        mock_tools = [{"name": "search_course_content"}]

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "What is computer use?", tools=mock_tools, tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="What is computer use?"
        )

        # Verify final response
        assert result == "Based on the search, computer use is..."

        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_flow(self, mock_anthropic_class):
        """Test the detailed flow of tool execution"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Setup tool use response
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [MagicMock()]

        tool_content = mock_tool_response.content[0]
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.id = "tool_123"
        tool_content.input = {"query": "test query"}

        # Setup final response
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "Final answer"

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator._handle_tool_execution(
            mock_tool_response,
            {
                "messages": [{"role": "user", "content": "test"}],
                "system": "system prompt",
            },
            mock_tool_manager,
        )

        # Verify tool execution
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test query"
        )

        # Verify second API call structure
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        assert (
            len(second_call_args["messages"]) == 3
        )  # Original + assistant + tool result

        assert result == "Final answer"

    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        assert "course materials and educational content" in AIGenerator.SYSTEM_PROMPT
        assert "Content Search Tool" in AIGenerator.SYSTEM_PROMPT
        assert "Course Outline Tool" in AIGenerator.SYSTEM_PROMPT
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT

    @patch("ai_generator.anthropic.Anthropic")
    def test_api_error_handling(self, mock_anthropic_class):
        """Test handling of API errors"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Mock API exception
        import anthropic

        mock_client.messages.create.side_effect = anthropic.APIError("API Error")

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")

        with pytest.raises(anthropic.APIError):
            generator.generate_response("Test query")

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_with_multiple_tools(self, mock_anthropic_class):
        """Test execution when multiple tools are called in one response"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Response with multiple tool calls
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = []

        # First tool call
        tool1 = MagicMock()
        tool1.type = "tool_use"
        tool1.name = "search_course_content"
        tool1.id = "tool1"
        tool1.input = {"query": "first query"}

        # Second tool call
        tool2 = MagicMock()
        tool2.type = "tool_use"
        tool2.name = "get_course_outline"
        tool2.id = "tool2"
        tool2.input = {"course_name": "Test Course"}

        mock_tool_response.content = [tool1, tool2]

        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "Combined results"

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Search result", "Outline result"]

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator._handle_tool_execution(
            mock_tool_response,
            {"messages": [{"role": "user", "content": "test"}], "system": "system"},
            mock_tool_manager,
        )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2

        # Check tool calls
        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0] == ("search_course_content",)
        assert calls[0][1] == {"query": "first query"}

        assert calls[1][0] == ("get_course_outline",)
        assert calls[1][1] == {"course_name": "Test Course"}

        assert result == "Combined results"

    @patch("ai_generator.anthropic.Anthropic")
    def test_empty_tool_results(self, mock_anthropic_class):
        """Test handling when tool returns empty results"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [MagicMock()]

        tool_content = mock_tool_response.content[0]
        tool_content.type = "tool_use"
        tool_content.name = "search_course_content"
        tool_content.id = "tool_123"
        tool_content.input = {"query": "nonexistent content"}

        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "No results found."

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Mock tool manager returning empty result (current system behavior)
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "No relevant content found."

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator._handle_tool_execution(
            mock_tool_response,
            {"messages": [{"role": "user", "content": "test"}], "system": "system"},
            mock_tool_manager,
        )

        # This test reveals what happens in the current failing scenario
        assert result == "No results found."


class TestSequentialToolCalling:
    """Test cases for sequential tool calling functionality"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_single_round_termination(self, mock_anthropic_class):
        """Test that single tool use terminates correctly"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: Tool use requested
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [MagicMock()]
        mock_tool_response.content[0].type = "tool_use"
        mock_tool_response.content[0].name = "search_course_content"
        mock_tool_response.content[0].id = "tool_123"
        mock_tool_response.content[0].input = {"query": "test"}

        # Round 2: Final response (no more tools)
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "Final answer after tool use."
        mock_final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Verify single round termination
        assert result == "Final answer after tool use."
        assert mock_client.messages.create.call_count == 2  # Tool round + final round

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_round_execution(self, mock_anthropic_class):
        """Test full two-round execution"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: Tool use
        mock_tool_response_1 = MagicMock()
        mock_tool_response_1.stop_reason = "tool_use"
        mock_tool_response_1.content = [MagicMock()]
        mock_tool_response_1.content[0].type = "tool_use"
        mock_tool_response_1.content[0].name = "get_course_outline"
        mock_tool_response_1.content[0].id = "tool_1"
        mock_tool_response_1.content[0].input = {"course_name": "Test Course"}

        # Round 2: Another tool use
        mock_tool_response_2 = MagicMock()
        mock_tool_response_2.stop_reason = "tool_use"
        mock_tool_response_2.content = [MagicMock()]
        mock_tool_response_2.content[0].type = "tool_use"
        mock_tool_response_2.content[0].name = "search_course_content"
        mock_tool_response_2.content[0].id = "tool_2"
        mock_tool_response_2.content[0].input = {"query": "lesson 4 topic"}

        # Final synthesis round
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "Combined results from both tools."

        mock_client.messages.create.side_effect = [
            mock_tool_response_1,
            mock_tool_response_2,
            mock_final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline result",
            "Content search result",
        ]

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Find course with same topic as lesson 4 of Test Course",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Verify two-round execution + final synthesis
        assert result == "Combined results from both tools."
        assert mock_client.messages.create.call_count == 3  # 2 tool rounds + final
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_rounds_termination(self, mock_anthropic_class):
        """Test termination after 2 rounds regardless of continued tool requests"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: Tool use
        mock_tool_response_1 = MagicMock()
        mock_tool_response_1.stop_reason = "tool_use"
        mock_tool_response_1.content = [MagicMock()]
        mock_tool_response_1.content[0].type = "tool_use"
        mock_tool_response_1.content[0].name = "search_course_content"
        mock_tool_response_1.content[0].id = "tool_1"
        mock_tool_response_1.content[0].input = {"query": "test1"}

        # Round 2: Another tool use (still requesting more)
        mock_tool_response_2 = MagicMock()
        mock_tool_response_2.stop_reason = "tool_use"  # Still wants more tools
        mock_tool_response_2.content = [MagicMock()]
        mock_tool_response_2.content[0].type = "tool_use"
        mock_tool_response_2.content[0].name = "search_course_content"
        mock_tool_response_2.content[0].id = "tool_2"
        mock_tool_response_2.content[0].input = {"query": "test2"}

        # Final synthesis (forced without tools)
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "Final synthesis after max rounds."

        mock_client.messages.create.side_effect = [
            mock_tool_response_1,
            mock_tool_response_2,
            mock_final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Complex query needing multiple searches",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Should terminate after 2 rounds + final synthesis
        assert result == "Final synthesis after max rounds."
        assert mock_client.messages.create.call_count == 3

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_error_handling(self, mock_anthropic_class):
        """Test graceful handling of tool execution errors"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [MagicMock()]
        mock_tool_response.content[0].type = "tool_use"
        mock_tool_response.content[0].name = "search_course_content"
        mock_tool_response.content[0].id = "tool_123"
        mock_tool_response.content[0].input = {"query": "test"}

        # Final response after error
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "Handled error gracefully."
        mock_final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Mock tool execution error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Should handle error and continue
        assert result == "Handled error gracefully."

        # Verify error was included in tool result
        call_args = mock_client.messages.create.call_args_list[1]
        messages = call_args[1]["messages"]
        tool_result_message = messages[2]["content"][0]
        assert "Tool execution error: Tool failed" in tool_result_message["content"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_fallback_behavior(self, mock_anthropic_class):
        """Test fallback when API calls fail"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First call fails, fallback succeeds
        mock_client.messages.create.side_effect = [
            Exception("API Error"),
            MagicMock(content=[MagicMock(text="Fallback response")]),
        ]

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response("Test query")

        # Should use fallback
        assert result == "Fallback response"

    @patch("ai_generator.anthropic.Anthropic")
    def test_conversation_history_preservation(self, mock_anthropic_class):
        """Test that conversation history is maintained across rounds"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Tool response
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [MagicMock()]
        mock_tool_response.content[0].type = "tool_use"
        mock_tool_response.content[0].name = "search_course_content"
        mock_tool_response.content[0].id = "tool_123"
        mock_tool_response.content[0].input = {"query": "test"}

        # Final response
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "Final response"
        mock_final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response(
            "Test query",
            conversation_history="Previous conversation",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Verify history included in both API calls
        calls = mock_client.messages.create.call_args_list
        for call in calls:
            system_content = call[1]["system"]
            assert "Previous conversation" in system_content

        assert result == "Final response"
