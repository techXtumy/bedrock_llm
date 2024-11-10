import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc import ABC
from src.bedrock_llm.models.base import BaseModelImplementation
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.schema.message import MessageBlock
from src.bedrock_llm.types.enums import StopReason

class TestBaseModelImplementation(unittest.TestCase):

    def test_abstract_methods(self):
        # Ensure that BaseModelImplementation is an abstract base class
        self.assertTrue(issubclass(BaseModelImplementation, ABC))
        
        # Check that abstract methods are defined
        abstract_methods = [
            'prepare_request',
            'prepare_request_async',
            'parse_response',
            'parse_stream_response'
        ]
        for method in abstract_methods:
            self.assertTrue(hasattr(BaseModelImplementation, method))
            self.assertTrue(callable(getattr(BaseModelImplementation, method)))

    def test_concrete_implementation(self):
        # Define a concrete implementation of BaseModelImplementation
        class ConcreteModelImplementation(BaseModelImplementation):
            def prepare_request(self, config, prompt, system=None, tools=None, **kwargs):
                return {"request": "prepared"}

            async def prepare_request_async(self, config, prompt, system=None, tools=None, **kwargs):
                return {"request": "prepared asynchronously"}

            def parse_response(self, response):
                return MessageBlock(role="assistant", content="Parsed response"), StopReason.END_TURN

            async def parse_stream_response(self, response):
                yield MessageBlock(role="assistant", content="Parsed stream response"), StopReason.END_TURN

        # Instantiate the concrete implementation
        concrete_model = ConcreteModelImplementation()

        # Test prepare_request
        config = ModelConfig()
        prompt = "Test prompt"
        request = concrete_model.prepare_request(config, prompt)
        self.assertEqual(request, {"request": "prepared"})

        # Test parse_response
        response, stop_reason = concrete_model.parse_response("Sample response")
        self.assertIsInstance(response, MessageBlock)
        self.assertEqual(response.role, "assistant")
        self.assertEqual(response.content, "Parsed response")
        self.assertEqual(stop_reason, StopReason.END_TURN)

        # Note: We can't easily test async methods in a synchronous unit test framework
        # For a complete test suite, you would need to use an async test runner or framework

if __name__ == '__main__':
    unittest.main()