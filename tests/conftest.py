"""Common test fixtures and configurations."""
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_bedrock_client():
    """Create a mock Bedrock client."""
    client = MagicMock()
    client.invoke_model.return_value = {
        'body': MagicMock(),
        'contentType': 'application/json',
        'statusCode': 200,
    }
    return client

@pytest.fixture
def mock_response_stream():
    """Create a mock response stream."""
    def mock_stream():
        yield {
            'chunk': {
                'bytes': b'{"completion": "Test response", "stop_reason": "stop"}'
            }
        }
    return mock_stream
