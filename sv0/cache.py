from collections import deque

class BoundedLRUCache:
    """Memory-bounded LRU cache"""
    def __init__(self, max_size: int):
        self.cache = {}
        self.max_size = int(max_size)
        self.access_order = deque(maxlen=max_size)

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        if key in self.cache:
            # Move to end (most recent)
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
            self.access_order.append(key)
            return self.cache[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key not in self.cache and len(self.cache) >= self.max_size:
            # Evict least recently used
            oldest = self.access_order[0]
            self.cache.pop(oldest, None)
        self.cache[key] = value
        try:
            self.access_order.remove(key)
        except ValueError:
            pass
        self.access_order.append(key)
