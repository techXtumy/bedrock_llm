import unittest
import asyncio
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unittest.mock import patch, MagicMock, AsyncMock
from src.bedrock_llm.agent import Agent
from src.bedrock_llm.types.enums import ModelName, StopReason
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.schema.message import MessageBlock, ToolUseBlock, ToolResultBlock
from src.bedrock_llm.schema.tools import ToolMetadata

class TestAgent(unittest.TestCase):

    def setUp(self):
        self.region_name = "us-west-2"
        self.model_name = ModelName.CLAUDE_3_HAIKU
        self.max_iterations = 3

    @patch('src.bedrock_llm.agent.LLMClient.__init__')
    def test_agent_initialization(self, mock_llm_client_init):
        mock_llm_client_init.return_value = None
        agent = Agent(self.region_name, self.model_name, max_iterations=self.max_iterations)
        self.assertEqual(agent.region_name, self.region_name)
        self.assertEqual(agent.model_name, self.model_name)
        self.assertEqual(agent.max_iterations, self.max_iterations)
        mock_llm_client_init.assert_called_once_with(self.region_name, self.model_name, [], None)

    def test_tool_decorator(self):
        agent = Agent(self.region_name, self.model_name)
        
        @agent.tool(ToolMetadata(name="test_tool", description="A test tool"))
        def test_tool(arg1: str, arg2: int):
            return f"Result: {arg1}, {arg2}"

        self.assertIn("test_tool", agent.tool_functions)
        self.assertEqual(agent.tool_functions["test_tool"]["metadata"].name, "test_tool")
        self.assertEqual(agent.tool_functions["test_tool"]["metadata"].description, "A test tool")
        
        result = agent.tool_functions["test_tool"]["function"]("hello", 42)
        self.assertEqual(result, "Result: hello, 42")

    @patch('src.bedrock_llm.agent.Agent.generate')
    def test_generate_and_action_async(self, mock_generate):
        async def run_test():
            agent = Agent(self.region_name, self.model_name)
            
            # Mock the generate method to simulate different scenarios
            mock_generate.side_effect = [
                (MessageBlock(role="assistant", content="Using tool: test_tool"), StopReason.TOOL_USE),
                (MessageBlock(role="assistant", content="Final response"), StopReason.END_TURN),
            ]
            
            # Add a mock tool
            @agent.tool(ToolMetadata(name="test_tool", description="A test tool"))
            def test_tool(arg1: str):
                return f"Tool result: {arg1}"
            
            prompt = "Test prompt"
            tools = ["test_tool"]
            config = ModelConfig()
            
            results = []
            async for token, stop_reason, response, tool_result in agent.generate_and_action_async(prompt, tools, config=config):
                results.append((token, stop_reason, response, tool_result))
            
            # Check the results
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0], (None, StopReason.TOOL_USE, MessageBlock(role="assistant", content="Using tool: test_tool"), None))
            self.assertEqual(results[1], (None, None, None, ["Tool result: test_tool"]))
            self.assertEqual(results[2], (None, StopReason.END_TURN, MessageBlock(role="assistant", content="Final response"), None))

        asyncio.run(run_test())

    @patch('src.bedrock_llm.agent.Agent.generate')
    def test_generate_and_action_async_max_iterations(self, mock_generate):
        async def run_test():
            agent = Agent(self.region_name, self.model_name, max_iterations=2)
            
            # Mock the generate method to simulate exceeding max iterations
            mock_generate.side_effect = [
                (MessageBlock(role="assistant", content="Using tool: test_tool"), StopReason.TOOL_USE),
                (MessageBlock(role="assistant", content="Using tool: test_tool"), StopReason.TOOL_USE),
                (MessageBlock(role="assistant", content="This should not be reached"), StopReason.END_TURN),
            ]
            
            # Add a mock tool
            @agent.tool(ToolMetadata(name="test_tool", description="A test tool"))
            def test_tool(arg1: str):
                return f"Tool result: {arg1}"
            
            prompt = "Test prompt"
            tools = ["test_tool"]
            config = ModelConfig()
            
            results = []
            async for token, stop_reason, response, tool_result in agent.generate_and_action_async(prompt, tools, config=config):
                results.append((token, stop_reason, response, tool_result))
            
            # Check the results
            self.assertEqual(len(results), 4)
            self.assertEqual(results[0], (None, StopReason.TOOL_USE, MessageBlock(role="assistant", content="Using tool: test_tool"), None))
            self.assertEqual(results[1], (None, None, None, ["Tool result: test_tool"]))
            self.assertEqual(results[2], (None, StopReason.TOOL_USE, MessageBlock(role="assistant", content="Using tool: test_tool"), None))
            self.assertEqual(results[3], (None, None, None, ["Tool result: test_tool"]))

        asyncio.run(run_test())

    def test_tool_decorator_with_invalid_metadata(self):
        agent = Agent(self.region_name, self.model_name)
        
        with self.assertRaises(ValueError):
            @agent.tool("Invalid metadata")
            def invalid_tool():
                pass

    def test_tool_decorator_with_duplicate_name(self):
        agent = Agent(self.region_name, self.model_name)
        
        @agent.tool(ToolMetadata(name="duplicate_tool", description="First tool"))
        def first_tool():
            pass

        with self.assertRaises(ValueError):
            @agent.tool(ToolMetadata(name="duplicate_tool", description="Second tool"))
            def second_tool():
                pass

if __name__ == '__main__':
    unittest.main()