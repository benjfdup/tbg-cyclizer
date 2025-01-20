from typing import Dict

class IndexesMethodPair():
    def __init__(self, indexes: Dict[str, int], method: str):
        '''
        A wrapper to help with organization of indexes, method pairings.
        '''
    
        self.indexes = indexes
        self.method = method