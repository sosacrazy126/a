import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from gptme.memory import MemoryManager, MemoryMode, Message

@pytest.fixture
def memory_client():
    with patch('gptme.memory.AsyncMemoryClient') as mock:
        client = AsyncMock()
        client.add = AsyncMock()
        client.search = AsyncMock()
        client.delete = AsyncMock()
        mock.return_value = client
        yield client

@pytest.fixture
def memory_manager(memory_client):
    context = {
        "user_id": "test_user",
        "agent_id": "test_agent",
        "run_id": "test_run"
    }
    return MemoryManager(
        api_key="test_key",
        mode=MemoryMode.FULL,
        context=context
    )

@pytest.fixture
def test_message():
    return Message(
        role="user",
        content="test message",
        timestamp=datetime.fromisoformat("2024-01-01T12:00:00")
    )

@pytest.mark.asyncio
async def test_store_message(memory_manager, memory_client, test_message):
    # Test storing a message
    await memory_manager.store_message(test_message)
    
    memory_client.add.assert_called_once_with(
        [{
            "role": "user",
            "content": "test message",
            "metadata": {
                "timestamp": "2024-01-01T12:00:00",
                "user_id": "test_user",
                "agent_id": "test_agent",
                "run_id": "test_run"
            }
        }],
        user_id="test_user",
        agent_id="test_agent",
        run_id="test_run"
    )

@pytest.mark.asyncio
async def test_get_relevant_context(memory_manager, memory_client):
    # Mock search results
    search_results = [
        {
            "id": "1",
            "text": "test message 1",
            "metadata": {
                "role": "user",
                "timestamp": "2024-01-01T12:00:00",
            }
        },
        {
            "id": "2", 
            "text": "test message 2",
            "metadata": {
                "role": "assistant",
                "timestamp": "2024-01-01T12:01:00",
            }
        }
    ]
    memory_client.search.return_value = search_results
    
    # Test retrieving context
    messages = await memory_manager.get_relevant_context("test query")
    
    assert len(messages) == 2
    assert messages[0].content == "test message 1"
    assert messages[0].role == "user"
    assert messages[1].content == "test message 2"
    assert messages[1].role == "assistant"
    
    memory_client.search.assert_called_once_with(
        "test query",
        user_id="test_user",
        agent_id="test_agent",
        run_id="test_run",
        limit=5
    )

@pytest.mark.asyncio
async def test_remove_message(memory_manager, memory_client, test_message):
    # Mock search results for removal
    search_results = [{
        "id": "1",
        "text": "test message",
        "metadata": {
            "role": "user",
            "timestamp": "2024-01-01T12:00:00",
        }
    }]
    memory_client.search.return_value = search_results
    
    # Test removing a message
    await memory_manager.remove_message(test_message)
    
    memory_client.delete.assert_called_once_with("1")

@pytest.mark.asyncio
async def test_disabled_mode():
    # Test with disabled memory mode
    memory_manager = MemoryManager(mode=MemoryMode.DISABLED)
    test_message = Message(
        role="user",
        content="test message",
        timestamp=datetime.now()
    )
    
    # All operations should return immediately
    await memory_manager.store_message(test_message)
    messages = await memory_manager.get_relevant_context("test")
    await memory_manager.remove_message(test_message)
    
    assert messages == []

@pytest.mark.asyncio
async def test_error_handling(memory_manager, memory_client):
    # Test error handling for store_message
    memory_client.add.side_effect = Exception("Test error")
    test_message = Message(role="user", content="test", timestamp=datetime.now())
    await memory_manager.store_message(test_message)  # Should not raise
    
    # Test error handling for get_relevant_context
    memory_client.search.side_effect = Exception("Test error")
    messages = await memory_manager.get_relevant_context("test")
    assert messages == []
    
    # Test error handling for remove_message
    memory_client.delete.side_effect = Exception("Test error")
    await memory_manager.remove_message(test_message)  # Should not raise

@pytest.mark.asyncio
async def test_json_parsing(memory_manager, memory_client):
    # Test handling of string JSON responses
    json_response = '[{"id":"1","text":"test","metadata":{"role":"user","timestamp":"2024-01-01T12:00:00"}}]'
    memory_client.search.return_value = json_response
    
    messages = await memory_manager.get_relevant_context("test")
    assert len(messages) == 1
    assert messages[0].content == "test"
    assert messages[0].role == "user"

@pytest.mark.asyncio
async def test_empty_messages(memory_manager, memory_client):
    # Test handling of empty messages
    search_results = [
        {
            "id": "1",
            "text": "",  # Empty text
            "metadata": {
                "role": "user",
                "timestamp": "2024-01-01T12:00:00",
            }
        },
        {
            "id": "2",
            "text": "valid message",
            "metadata": {
                "role": "assistant",
                "timestamp": "2024-01-01T12:01:00",
            }
        }
    ]
    memory_client.search.return_value = search_results
    
    messages = await memory_manager.get_relevant_context("test")
    assert len(messages) == 1  # Only the valid message should be included
    assert messages[0].content == "valid message" 