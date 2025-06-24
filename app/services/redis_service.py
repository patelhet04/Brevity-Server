# services/redis_service.py
import redis
import json
from typing import Optional, Dict, Any
from datetime import datetime
from app.config import settings


class RedisSessionStore:
    def __init__(self):
        try:
            self.redis = redis.from_url(
                settings.redis_url, decode_responses=True)
            self.session_ttl = settings.session_ttl

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


# Create global instance
redis_store = RedisSessionStore()
