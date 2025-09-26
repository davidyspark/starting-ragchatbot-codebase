import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **Content Search Tool** (`search_course_content`): Search within course materials for specific educational content
2. **Course Outline Tool** (`get_course_outline`): Get course overview including title, course link, instructor, and complete lesson list with lesson numbers and titles

Tool Usage Guidelines:
- **Sequential tool use**: You may use tools across multiple rounds of reasoning to gather comprehensive information
- **Maximum 2 rounds**: You have up to 2 opportunities to use tools before providing your final response
- **Strategic tool selection**:
  - First round: Use tools to gather initial information
  - Second round (if needed): Use additional tools to fill gaps or get complementary information
- **Synthesis**: After tool usage, synthesize results into accurate, fact-based responses

Tool Execution Rules:
- Each round allows multiple tool calls if needed
- Consider tool results when deciding whether additional tools are needed
- If you have sufficient information after one round, proceed directly to your final answer
- Use course outline tool for structural questions, content search for detailed material questions

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Complex course questions**: Use tools strategically across rounds as needed
- **No meta-commentary**: Provide direct answers only — no reasoning process, search explanations, or question-type analysis

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: str | None = None,
        tools: list | None = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with up to 2 rounds of sequential tool usage.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        try:
            # Build system content efficiently
            system_content = (
                f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
                if conversation_history
                else self.SYSTEM_PROMPT
            )

            # Initialize conversation for multi-round processing
            messages = [{"role": "user", "content": query}]

            # Process up to 2 rounds
            for round_num in range(1, 3):  # 1 and 2
                response = self._execute_round(
                    messages, system_content, tools, tool_manager, round_num
                )

                # If we got a final text response, return it
                if isinstance(response, str):
                    return response

                # Otherwise, response is the API response with tool use
                # Add it to messages and continue to next round
                messages.append({"role": "assistant", "content": response.content})

                # Execute tools and add results
                tool_results = self._execute_tools(response, tool_manager)
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # No valid tool results, end with error message
                    return "I encountered an error while processing your request."

            # If we reach here, we've completed 2 rounds with tools
            # Make one final call without tools for synthesis
            return self._execute_final_round(messages, system_content)

        except Exception as e:
            # Fallback to simple response without tools
            return self._fallback_response(query, conversation_history, str(e))

    def _execute_round(
        self,
        messages: list[dict],
        system_content: str,
        tools: list | None,
        tool_manager,
        round_num: int,
    ):
        """
        Execute a single round of the conversation.

        Args:
            messages: Current message history
            system_content: System prompt content
            tools: Available tools
            tool_manager: Tool execution manager
            round_num: Current round number (1 or 2)

        Returns:
            - str: Final text response (terminates sequence)
            - anthropic.Message: Response with tool_use (continues sequence)
        """
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        # Add tools if available and we haven't exceeded max rounds
        if tools and round_num <= 2:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        response = self.client.messages.create(**api_params)

        # Check for tool use
        if response.stop_reason == "tool_use" and tool_manager:
            return response  # Continue to next round

        # Return final text response
        return response.content[0].text

    def _execute_tools(self, response, tool_manager) -> list[dict] | None:
        """
        Execute all tool calls and return formatted results.

        Args:
            response: API response containing tool use requests
            tool_manager: Tool execution manager

        Returns:
            List of tool results or None if no tools to execute
        """
        if response.stop_reason != "tool_use":
            return None

        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Handle tool execution errors gracefully
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution error: {str(e)}",
                        }
                    )

        return tool_results if tool_results else None

    def _execute_final_round(self, messages: list[dict], system_content: str) -> str:
        """
        Execute final round without tools for response synthesis.

        Args:
            messages: Complete message history
            system_content: System prompt content

        Returns:
            Final synthesized response
        """
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
            # Deliberately no tools - force final synthesis
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _fallback_response(
        self,
        query: str,
        conversation_history: str | None = None,
        error: str | None = None,
    ) -> str:
        """
        Fallback response generation without tools.

        Args:
            query: Original user query
            conversation_history: Previous conversation context
            error: Error message if available

        Returns:
            Fallback response
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        try:
            response = self.client.messages.create(
                **self.base_params,
                messages=[{"role": "user", "content": query}],
                system=system_content,
                # No tools - simple response only
            )
            return response.content[0].text
        except Exception:
            return "I encountered an error processing your request. Please try again."
