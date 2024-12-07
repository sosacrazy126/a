import os
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
from mem0 import AsyncMemoryClient
from .message import Message

logger = logging.getLogger(__name__)

class MemoryMode:
    """Memory operation modes."""
    FULL = "full"         # Mem0 API
    LOCAL = "local"       # Local cache only (not implemented)
    DISABLED = "disabled" # No memory features

class MemoryError(Exception):
    """Base class for memory-related errors."""
    pass

class MemoryConnectionError(MemoryError):
    """Raised when connection to Mem0 fails."""
    pass

class MemoryTimeoutError(MemoryError):
    """Raised when memory operations timeout."""
    pass

class MemoryManager:
    """Manages memory operations using Mem0."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        mode: str = MemoryMode.FULL,
        context: Optional[Dict[str, Any]] = None,
        fallback_threshold: int = 3
    ):
        """Initialize MemoryManager.
        
        Args:
            api_key: Mem0 API key. If not provided, will check MEM0_API_KEY env var
            mode: Operation mode (FULL, LOCAL, or DISABLED)
            context: Additional context for memory operations
            fallback_threshold: Number of failures before falling back to LOCAL mode
        """
        self.mode = mode
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        
        if self.mode == MemoryMode.FULL and not self.api_key:
            raise ValueError("API key required for full mode")
            
        self.context = context or {
            "user_id": "default_user",
            "agent_id": "default_agent",
            "run_id": f"run_{int(time.time())}"
        }
        
        self.client = self._initialize_client()
        self._failure_count = 0
        self.fallback_threshold = fallback_threshold
        
    def _initialize_client(self) -> Optional[AsyncMemoryClient]:
        """Initialize Mem0 client."""
        if self.mode == MemoryMode.FULL:
            try:
                return AsyncMemoryClient(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Mem0 client: {e}")
                self._handle_failure(e)
                return None
        return None
        
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
            
    async def store_message(self, message: Message) -> None:
        """Store a message in memory."""
        if self.mode == MemoryMode.DISABLED:
            return
            
        if not self.validate_message(message):
            logger.warning("Invalid message format, skipping storage")
            return
            
        try:
            if self.mode == MemoryMode.FULL and self.client:
                # Format message for API
                messages = [{
                    "role": message.role,
                    "content": message.content,
                    "metadata": {
                        "timestamp": message.timestamp.isoformat(),
                        **self.context
                    }
                }]
                
                # Add message with metadata
                await self.client.add(
                    messages,
                    user_id=self.context.get("user_id", "default_user"),
                    agent_id=self.context.get("agent_id", "default_agent"),
                    run_id=self.context.get("run_id")
                )
                logger.info(f"Stored message: {message.content}")
                
        except Exception as e:
            logger.warning(f"Failed to store message: {e}")
            self._handle_failure(e)
            
    async def get_relevant_context(
        self,
        query: str,
        limit: int = 5
    ) -> list[Message]:
        """Retrieve relevant messages based on query."""
        if self.mode == MemoryMode.DISABLED:
            return []
            
        try:
            messages = []
            
            if self.mode == MemoryMode.FULL and self.client:
                # Search with basic filters
                search_results = await self.client.search(
                    query,
                    user_id=self.context.get("user_id", "default_user"),
                    agent_id=self.context.get("agent_id", "default_agent"),
                    run_id=self.context.get("run_id"),
                    limit=limit
                )
                
                # Handle both string and dict responses
                if isinstance(search_results, str):
                    try:
                        import json
                        results = json.loads(search_results)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse search results JSON")
                        return []
                else:
                    results = search_results
                
                # Ensure results is a list
                if not isinstance(results, list):
                    results = [results]
                
                # Convert results to messages
                for result in results:
                    try:
                        # Handle different response formats
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except json.JSONDecodeError:
                                continue
                        
                        # Extract message data
                        memory_data = result.get("memory", result)  # Fallback to result if memory key doesn't exist
                        
                        # Skip empty messages
                        if not memory_data.get("content", "").strip():
                            continue
                            
                        messages.append(Message(
                            role=memory_data.get("role", "user"),
                            content=memory_data.get("content", ""),
                            timestamp=datetime.fromisoformat(
                                memory_data.get("metadata", {}).get("timestamp", datetime.now().isoformat())
                            )
                        ))
                    except (KeyError, ValueError, AttributeError) as e:
                        logger.warning(f"Failed to parse result: {e}")
                        continue
            
            # Sort by timestamp
            messages.sort(key=lambda x: x.timestamp)
            return messages[:limit]
                
        except Exception as e:
            logger.warning(f"Failed to retrieve context: {e}")
            self._handle_failure(e)
            return []
            
    async def clear(self) -> None:
        """Clear all stored memories."""
        try:
            if self.mode == MemoryMode.FULL and self.client:
                # Delete all memories for this context
                await self.client.delete_all(
                    user_id=self.context.get("user_id", "default_user"),
                    agent_id=self.context.get("agent_id", "default_agent"),
                    run_id=self.context.get("run_id")
                )
                logger.info("Cleared all memories.")
        except Exception as e:
            logger.warning(f"Failed to clear memory: {e}")
            self._handle_failure(e)
                
    async def remove_message(self, message: Message) -> None:
        """Remove a message from memory."""
        if self.mode == MemoryMode.DISABLED:
            return
            
        try:
            if self.mode == MemoryMode.FULL and self.client:
                # Search for the specific message
                search_results = await self.client.search(
                    message.content,  # Use content as search query
                    user_id=self.context.get("user_id", "default_user"),
                    agent_id=self.context.get("agent_id", "default_agent"),
                    run_id=self.context.get("run_id"),
                    limit=10  # Get a few matches to find the exact one
                )
                
                # Handle both string and dict responses
                if isinstance(search_results, str):
                    try:
                        import json
                        results = json.loads(search_results)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse search results JSON")
                        return
                else:
                    results = search_results
                
                # Ensure results is a list
                if not isinstance(results, list):
                    results = [results]
                
                # Find and delete the exact message
                for result in results:
                    try:
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except json.JSONDecodeError:
                                continue
                                
                        text = result.get("text", "").strip()
                        metadata = result.get("metadata", {})
                        
                        if (
                            text == message.content
                            and metadata.get("role") == message.role
                            and metadata.get("timestamp") == message.timestamp.isoformat()
                        ):
                            memory_id = result.get("id")
                            if memory_id:
                                await self.client.delete(memory_id)
                                logger.info(f"Removed message with ID: {memory_id}")
                                break
                    except (KeyError, ValueError, AttributeError) as e:
                        logger.warning(f"Failed to parse result for removal: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Failed to remove message: {e}")
            self._handle_failure(e)
            
    def validate_message(self, message: Message) -> bool:
        """Validate message before storage."""
        try:
            # Basic validation
            if not message.content.strip():
                logger.warning("Empty message content")
                return False
                
            if message.role not in ["user", "assistant", "system"]:
                logger.warning(f"Invalid role: {message.role}")
                return False
                
            # Ensure timestamp exists and can be formatted
            if not message.timestamp:
                logger.warning("Missing timestamp")
                return False
                
            message.timestamp.isoformat()
            return True
            
        except Exception as e:
            logger.warning(f"Message validation failed: {e}")
            return False