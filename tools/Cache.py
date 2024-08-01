import pickle

import requests
from cachetools import cached, TTLCache

cache_filename = 'cache.pkl'  # 缓存文件名
# 尝试从缓存文件中加载缓存数据
try:
    with open(cache_filename, 'rb') as file:
        cache = pickle.load(file)
except FileNotFoundError:
    cache = TTLCache(maxsize=1000, ttl=360000)  # 创建新的缓存对象

@cached(cache)
def make_cached_request(url):
    response = requests.get(url)
    return response.json()
def dump_cache(response):
    with open(cache_filename, 'wb') as file:
        pickle.dump(response, file)