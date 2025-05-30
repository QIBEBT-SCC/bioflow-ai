import redis

from app.core.config import get_settings

monitor_redis_client = redis.Redis.from_url(get_settings().persistent.MONITOR_CACHE)
