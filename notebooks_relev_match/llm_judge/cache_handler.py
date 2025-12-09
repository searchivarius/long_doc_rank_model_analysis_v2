#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
import json

class CacheManagerGeneric:
    """
        A simple filesystem-based cache-managing class. It has get/put functions that create
        a cache entry given an ID. The put function has a mechanism to prevent writing incomplete
        cache entries: 
        1. First it writes all data to a temporary file. 
        2. Second it renames the temporary file into a permanent cache file.
        3. Temporary and permanent cache file have the same prefix that is defined by the cache ID.
        4. All cache files are stored inside a given directory.
    """    
    def __init__(self, cache_dir):
        """
            :param cache_dir: The directory where cache files are stored.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get(self, cache_id):
        """
            :param cache_id: The ID of the cache entry to be retrieved.
            :return: The cache entry data, or None if the cache entry does not exist.
        """
        cache_file = os.path.join(self.cache_dir, cache_id)
        if not os.path.exists(cache_file):
            return None
        with open(cache_file, 'r') as f:
            return f.read()
        
    def put(self, cache_id, data):
        """
            :param cache_id: The ID of the cache entry to be stored.
            :param data: The data to be stored in the cache entry.
            :return: None
        """        
        cache_file = os.path.join(self.cache_dir, cache_id)
        temp_file = cache_file + '.tmp_entry'
        with open(temp_file, 'w') as f:
            f.write(data)
        os.rename(temp_file, cache_file)


class CacheManagerJSON(CacheManagerGeneric):
    """
        A cache-manager subclass specializing on loading/storing JSON data.
    """    
    def __init__(self, cache_dir):
        super().__init__(cache_dir)

    def get(self, cache_id):
        data = super().get(cache_id)
        if data is None:
            return None
        return json.loads(data)

    def put(self, cache_id, data):
        super().put(cache_id, json.dumps(data))