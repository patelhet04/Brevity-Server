# services/redis_service.py
import redis
import json
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.config import settings


class RedisSessionStore:
    def __init__(self):
        try:
            self.redis = redis.from_url(
                settings.redis_url, decode_responses=True)
            self.session_ttl = settings.session_ttl

            # Conversation TTL - using same as session for now
            self.conversation_ttl = settings.session_ttl

            # Test connection
            self.redis.ping()
            print("✅ Connected to Redis")

        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            raise

    def get_session(self, session_id: str) -> Optional[Dict[Any, Any]]:
        """Get session data by session ID"""
        try:
            data = self.redis.get(f"session:{session_id}")
            return json.loads(data) if data else None
        except Exception as e:
            print(f"Redis get error: {e}")
            return None

    def set_session(self, session_id: str, data: Dict[Any, Any]) -> bool:
        """Set session data with TTL"""
        try:
            # Add timestamp
            data["last_activity"] = datetime.now().isoformat()

            # Store with expiry
            self.redis.setex(
                f"session:{session_id}",
                self.session_ttl,
                json.dumps(data, default=str)
            )
            return True

        except Exception as e:
            print(f"Redis set error: {e}")
            return False

    def extend_session(self, session_id: str) -> bool:
        """Extend session TTL"""
        try:
            return self.redis.expire(f"session:{session_id}", self.session_ttl)
        except Exception as e:
            print(f"Redis extend error: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        try:
            return self.redis.delete(f"session:{session_id}") > 0
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False

    def get_conversation(self, conversation_id: str) -> Optional[List[Dict[str, str]]]:
        """
        Get conversation history by conversation ID

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            List of chat messages or None if not found
        """
        try:
            data = self.redis.get(f"conversation:{conversation_id}")
            if data:
                conversation_data = json.loads(data)
                return conversation_data.get("history", [])
            return None
        except Exception as e:
            print(f"Redis conversation get error: {e}")
            return None

    def save_conversation(self, conversation_id: str, history: List[Dict[str, str]],
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save conversation history with TTL

        Args:
            conversation_id: Unique conversation identifier
            history: List of chat messages (role, content)
            metadata: Optional metadata about the conversation

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare conversation data
            conversation_data = {
                "history": history,
                "last_activity": datetime.now().isoformat(),
                "message_count": len(history),
                "created_at": metadata.get("created_at", datetime.now().isoformat()) if metadata else datetime.now().isoformat()
            }

            # Add any additional metadata
            if metadata:
                conversation_data.update(metadata)

            # Store with expiry
            self.redis.setex(
                f"conversation:{conversation_id}",
                self.conversation_ttl,
                json.dumps(conversation_data, default=str)
            )
            return True

        except Exception as e:
            print(f"Redis conversation save error: {e}")
            return False

    def extend_conversation(self, conversation_id: str) -> bool:
        """
        Extend conversation TTL (keep conversation alive)

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            return self.redis.expire(f"conversation:{conversation_id}", self.conversation_ttl)
        except Exception as e:
            print(f"Redis conversation extend error: {e}")
            return False

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete conversation

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            return self.redis.delete(f"conversation:{conversation_id}") > 0
        except Exception as e:
            print(f"Redis conversation delete error: {e}")
            return False

    def generate_conversation_id(self) -> str:
        """
        Generate a unique conversation ID

        Returns:
            UUID-based conversation ID
        """
        return f"conv_{uuid.uuid4().hex[:12]}"

    def conversation_exists(self, conversation_id: str) -> bool:
        """
        Check if conversation exists

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            True if conversation exists, False otherwise
        """
        try:
            return self.redis.exists(f"conversation:{conversation_id}") > 0
        except Exception as e:
            print(f"Redis conversation exists error: {e}")
            return False

    def get_conversation_metadata(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation metadata without full history

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            Conversation metadata or None if not found
        """
        try:
            data = self.redis.get(f"conversation:{conversation_id}")
            if data:
                conversation_data = json.loads(data)
                # Return everything except history
                metadata = {k: v for k, v in conversation_data.items()
                            if k != "history"}
                return metadata
            return None
        except Exception as e:
            print(f"Redis conversation metadata error: {e}")
            return None


# Create global instance
redis_store = RedisSessionStore()
