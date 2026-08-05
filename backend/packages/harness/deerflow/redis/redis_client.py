import json
import redis
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict, get_args, Optional, Dict

REDIS_CONF = {
    "host": "192.168.0.106",
    "port": 6379,
    "db": 0,
    "password": "mypassword",
    "decode_responses": True,
    "socket_timeout": 5
}

class _RedisClient:
    """内部私有Redis工具，不对外暴露"""
    def __init__(self, conf: dict):
        self.client = redis.Redis(**conf)

    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        return self.client.set(key, value, ex=expire)

    def get(self, key: str) -> Optional[Any]:
        raw = self.client.get(key)
        if raw is None:
            return None
        # 尝试反序列化json
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # 普通字符串直接返回
            return raw

    def delete(self, *keys: str) -> int:
        return self.client.delete(*keys)

    def exists(self, key: str) -> bool:
        return bool(self.client.exists(key))

    def set_hash(self, key: str, data: Dict, expire: Optional[int] = None):
        self.client.hset(key, mapping=data)
        if expire:
            self.client.expire(key, expire)

    def get_hash(self, key: str) -> Dict:
        return self.client.hgetall(key)

    def get_hash_field(self, key: str, field: str):
        return self.client.hget(key, field)

    def close(self):
        self.client.close()

class GlobalMemoryRedis1:
    """全局统一缓存入口类（单例）"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        # 单例模式，全局只创建一次
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, redis_conf: dict):
        if not hasattr(self, "redis"):
            # 内部持有redis实例
            self.redis = _RedisClient(redis_conf)

    # ========== 对外统一封装方法 ==========
    def set_cache(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """写入字符串缓存"""
        return self.redis.set(key, value, expire)

    def get_cache(self, key: str) -> Optional[Any]:
        """读取字符串缓存"""
        return self.redis.get(key)

    def del_cache(self, *keys: str):
        """删除缓存key"""
        return self.redis.delete(*keys)

    def has_key(self, key: str) -> bool:
        """判断key是否存在"""
        return self.redis.exists(key)

    def set_dict_cache(self, key: str, data: Dict, expire: Optional[int] = None):
        """字典哈希缓存"""
        self.redis.set_hash(key, data, expire)

    def get_dict_cache(self, key: str) -> Dict:
        """读取整个哈希字典"""
        return self.redis.get_hash(key)

    def get_dict_field(self, key: str, field: str):
        """读取哈希单个字段"""
        return self.redis.get_hash_field(key, field)

    def close(self):
        """关闭redis连接"""
        self.redis.close()


# 初始化全局单例，项目直接导入使用
# from config import REDIS_CONF
memoryRedis1 = GlobalMemoryRedis1(REDIS_CONF)
