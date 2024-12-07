# Mem0 Integration Notes

## Validated API Methods

### 1. Store Message
```python
# Correct implementation
async def store_message(self, message: Message) -> None:
    try:
        if self.mode == MemoryMode.FULL and self.client:
            await self.client.add(
                messages=[{
                    "text": message.content,
                    "metadata": {
                        "role": message.role,
                        "timestamp": message.timestamp.isoformat(),
                        **self.context
                    }
                }]
            )
    except Exception as e:
        logger.warning(f"Failed to store message: {e}")
        self._handle_failure(e)

# Error handling cases:
# 1. Network errors
# 2. API key validation
# 3. Rate limiting
# 4. Invalid message format
```

### 2. Search Messages
```python
# Correct implementation
async def get_relevant_context(self, query: str, limit: int = 5) -> list[Message]:
    try:
        if self.mode == MemoryMode.FULL and self.client:
            results = await self.client.search(
                text=query,
                metadata=self.context,
                limit=limit
            )
            
            return [
                Message(
                    role=result.metadata.get("role", "assistant"),
                    content=result.text,
                    timestamp=datetime.fromisoformat(
                        result.metadata.get("timestamp", datetime.now().isoformat())
                    )
                ) for result in results
            ]
    except Exception as e:
        logger.warning(f"Failed to retrieve context: {e}")
        self._handle_failure(e)
        return []

# Error handling cases:
# 1. Empty query
# 2. Invalid metadata filter
# 3. No results found
# 4. Malformed response
```

### 3. Clear Messages
```python
# Correct implementation
async def clear(self) -> None:
    try:
        if self.mode == MemoryMode.FULL and self.client:
            await self.client.delete_all(
                metadata=self.context
            )
    except Exception as e:
        logger.warning(f"Failed to clear memory: {e}")
        self._handle_failure(e)

# Error handling cases:
# 1. Permission denied
# 2. Invalid metadata filter
# 3. Partial deletion
```

### 4. Remove Message
```python
# Correct implementation
async def remove_message(self, message: Message) -> None:
    try:
        if self.mode == MemoryMode.FULL and self.client:
            await self.client.delete(
                metadata={
                    "role": message.role,
                    "timestamp": message.timestamp.isoformat(),
                    **self.context
                }
            )
    except Exception as e:
        logger.warning(f"Failed to remove message: {e}")
        self._handle_failure(e)

# Error handling cases:
# 1. Message not found
# 2. Invalid metadata format
# 3. Multiple matches
```

## Error Handling Strategy

### 1. Failure Detection
```python
def _handle_failure(self, error: Exception) -> None:
    """Handle operation failures."""
    self._failure_count += 1
    if self._failure_count >= self.fallback_threshold:
        prev_mode = self.mode
        self.mode = MemoryMode.LOCAL
        logger.warning(
            f"Too many failures ({self._failure_count}), "
            f"switching from {prev_mode} to {self.mode}"
        )
```

### 2. Recovery Mechanism
```python
async def check_connection(self) -> bool:
    """Check if connection to Mem0 is working."""
    try:
        # Try a simple operation
        await self.client.search(
            text="test",
            metadata=self.context,
            limit=1
        )
        return True
    except Exception:
        return False

async def attempt_recovery(self) -> bool:
    """Attempt to recover from failure state."""
    if await self.check_connection():
        self._failure_count = 0
        self.mode = MemoryMode.FULL
        logger.info("Successfully recovered memory connection")
        return True
    return False
```

### 3. Validation
```python
def validate_message(self, message: Message) -> bool:
    """Validate message before storage."""
    if not message.content.strip():
        return False
    if message.role not in ["user", "assistant", "system"]:
        return False
    try:
        message.timestamp.isoformat()
    except (AttributeError, ValueError):
        return False
    return True

def validate_metadata(self, metadata: dict) -> bool:
    """Validate metadata structure."""
    required_fields = ["role", "timestamp"]
    return all(field in metadata for field in required_fields)
```

## Testing Strategy

### 1. Unit Tests
```python
class TestMemoryOperations:
    async def test_store_message(self):
        manager = MemoryManager(api_key="test_key")
        message = Message(
            role="user",
            content="test message",
            timestamp=datetime.now()
        )
        await manager.store_message(message)
        
        # Verify storage
        results = await manager.get_relevant_context("test message")
        assert len(results) == 1
        assert results[0].content == message.content

    async def test_error_handling(self):
        manager = MemoryManager(api_key="invalid_key")
        message = Message(
            role="user",
            content="test message",
            timestamp=datetime.now()
        )
        
        # Should handle error and switch to LOCAL mode
        await manager.store_message(message)
        assert manager.mode == MemoryMode.LOCAL
        assert manager._failure_count == 1
```

### 2. Integration Tests
```python
class TestMemoryIntegration:
    async def test_chat_flow(self):
        manager = MemoryManager(api_key="test_key")
        
        # Store a sequence of messages
        messages = [
            Message("user", "What is Python?"),
            Message("assistant", "Python is a programming language."),
            Message("user", "Tell me more.")
        ]
        
        for msg in messages:
            await manager.store_message(msg)
        
        # Test context retrieval
        context = await manager.get_relevant_context("Python")
        assert len(context) > 0
        assert any("Python" in msg.content for msg in context)
```

## Configuration

### 1. Memory Settings
```python
@dataclass
class MemoryConfig:
    api_key: str
    mode: MemoryMode = MemoryMode.FULL
    fallback_threshold: int = 3
    context_window: int = 1000
    max_results: int = 5
    min_relevance: float = 0.7
```

### 2. Environment Variables
```bash
# Required
export MEM0_API_KEY="your-api-key"

# Optional
export MEM0_MODE="full"  # full, local, disabled
export MEM0_FALLBACK_THRESHOLD="3"
export MEM0_CONTEXT_WINDOW="1000"
```

## Usage Examples

### 1. Basic Usage
```python
# Initialize memory manager
memory = MemoryManager(
    api_key="your-api-key",
    mode=MemoryMode.FULL,
    context={
        "conversation_id": "123",
        "user_id": "user_123"
    }
)

# Store message
await memory.store_message(
    Message(
        role="user",
        content="Hello, how are you?",
        timestamp=datetime.now()
    )
)

# Get context
context = await memory.get_relevant_context(
    query="Hello",
    limit=5
)
```

### 2. Error Recovery
```python
# Handle connection issues
try:
    await memory.store_message(message)
except Exception:
    if memory.mode == MemoryMode.LOCAL:
        # Attempt recovery
        if await memory.attempt_recovery():
            # Retry operation
            await memory.store_message(message)
```