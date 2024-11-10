import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unittest.mock import patch, MagicMock
from src.bedrock_llm.client import LLMClient
from src.bedrock_llm.types.enums import ModelName, StopReason
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.schema.message import MessageBlock
from src.bedrock_llm.config.base import RetryConfig
from botocore.exceptions import ClientError, ReadTimeoutError

class TestLLMClient(unittest.TestCase):

    def setUp(self):
        self.region_name = "us-east-1"
        self.model_name = ModelName.CLAUDE_3_5_HAIKU
        self.retry_config = RetryConfig(max_retries=3, retry_delay=1, exponential_backoff=True)

    @patch('boto3.client')
    def test_llm_client_initialization(self, mock_boto3_client):
        client = LLMClient(self.region_name, self.model_name, retry_config=self.retry_config)
        self.assertEqual(client.region_name, self.region_name)
        self.assertEqual(client.model_name, self.model_name)
        self.assertEqual(client.retry_config, self.retry_config)
        self.assertIsNotNone(client.bedrock_client)
        self.assertIsNotNone(client.model_implementation)

    @patch('src.bedrock_llm.client.LLMClient._initialize_bedrock_client')
    @patch('src.bedrock_llm.client.LLMClient._get_model_implementation')
    def test_generate_method_success(self, mock_get_model_implementation, mock_initialize_bedrock_client):
        # Mock the model implementation
        mock_model = MagicMock()
        mock_get_model_implementation.return_value = mock_model

        # Mock the bedrock client
        mock_bedrock_client = MagicMock()
        mock_initialize_bedrock_client.return_value = mock_bedrock_client

        # Create an LLMClient instance
        client = LLMClient(self.region_name, self.model_name, retry_config=self.retry_config)

        # Set up the mock responses
        mock_model.prepare_request.return_value = {"mocked": "request"}
        mock_bedrock_client.invoke_model.return_value = {"body": b'{"mocked": "response"}'}
        mock_model.parse_response.return_value = (MessageBlock(role="assistant", content="Mocked response"), StopReason.END_TURN)

        # Test the generate method
        prompt = "Test prompt"
        config = ModelConfig()
        response, stop_reason = client.generate(prompt=prompt, config=config)

        # Assertions
        self.assertIsInstance(response, MessageBlock)
        self.assertEqual(response.role, "assistant")
        self.assertEqual(response.content, "Mocked response")
        self.assertEqual(stop_reason, StopReason.END_TURN)

        # Verify that the methods were called with the correct arguments
        mock_model.prepare_request.assert_called_once_with(config=config, prompt=prompt, system=None, documents=None, tools=None)
        mock_bedrock_client.invoke_model.assert_called_once_with(
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
            body='{"mocked": "request"}'
        )
        mock_model.parse_response.assert_called_once()

    @patch('src.bedrock_llm.client.LLMClient._initialize_bedrock_client')
    @patch('src.bedrock_llm.client.LLMClient._get_model_implementation')
    @patch('time.sleep')
    def test_generate_method_with_retries(self, mock_sleep, mock_get_model_implementation, mock_initialize_bedrock_client):
        mock_model = MagicMock()
        mock_get_model_implementation.return_value = mock_model
        mock_bedrock_client = MagicMock()
        mock_initialize_bedrock_client.return_value = mock_bedrock_client

        client = LLMClient(self.region_name, self.model_name, retry_config=self.retry_config)

        mock_model.prepare_request.return_value = {"mocked": "request"}
        mock_bedrock_client.invoke_model.side_effect = [
            ReadTimeoutError(endpoint_url="", operation_name=""),
            ClientError({"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "invoke_model"),
            {"body": b'{"mocked": "response"}'}
        ]
        mock_model.parse_response.return_value = (MessageBlock(role="assistant", content="Mocked response"), StopReason.END_TURN)

        prompt = "Test prompt"
        config = ModelConfig()
        response, stop_reason = client.generate(prompt=prompt, config=config)

        self.assertIsInstance(response, MessageBlock)
        self.assertEqual(response.content, "Mocked response")
        self.assertEqual(stop_reason, StopReason.END_TURN)
        self.assertEqual(mock_bedrock_client.invoke_model.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch('src.bedrock_llm.client.LLMClient._initialize_bedrock_client')
    @patch('src.bedrock_llm.client.LLMClient._get_model_implementation')
    def test_generate_method_max_retries_exceeded(self, mock_get_model_implementation, mock_initialize_bedrock_client):
        mock_model = MagicMock()
        mock_get_model_implementation.return_value = mock_model
        mock_bedrock_client = MagicMock()
        mock_initialize_bedrock_client.return_value = mock_bedrock_client

        client = LLMClient(self.region_name, self.model_name, retry_config=self.retry_config)

        mock_model.prepare_request.return_value = {"mocked": "request"}
        mock_bedrock_client.invoke_model.side_effect = ReadTimeoutError(endpoint_url="", operation_name="")

        prompt = "Test prompt"
        config = ModelConfig()

        with self.assertRaises(Exception) as context:
            client.generate(prompt=prompt, config=config)

        self.assertTrue("Max retries reached" in str(context.exception))
        self.assertEqual(mock_bedrock_client.invoke_model.call_count, self.retry_config.max_retries)

    def test_generate_method_with_invalid_prompt(self):
        client = LLMClient(self.region_name, self.model_name, retry_config=self.retry_config)
        client.memory = []  # Set memory to simulate enabled memory

        with self.assertRaises(ValueError) as context:
            client.generate(prompt="Invalid string prompt")

        self.assertTrue("If memory is set, prompt must be a MessageBlock or list of MessageBlock" in str(context.exception))

if __name__ == '__main__':
    unittest.main()