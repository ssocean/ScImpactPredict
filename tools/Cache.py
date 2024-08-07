import pickle

import requests
from cachetools import cached, TTLCache

cache_filename = 'cache.pkl'   
 
try:
    with open(cache_filename, 'rb') as file:
        cache = pickle.load(file)
except FileNotFoundError:
    cache = TTLCache(maxsize=1000, ttl=360000)   

@cached(cache)
def make_cached_request(url):
    response = requests.get(url)
    return response.json()
def dump_cache(response):
    with open(cache_filename, 'wb') as file:
        pickle.dump(response, file)