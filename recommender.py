from pathlib import Path
import numpy as np
##
from database import RecommenderDatabase


class Recommender():
    '''
    '''

    
    model_is_trained = False

    product_list_count = 0

    database = None

    def __init__(self, path:str=None, name:int=None) -> None:
        self.database = RecommenderDatabase(path, name)
        if self.database.get_features_count():
            self.model_is_trained = True

    def preprocess(self):
        pass

    def recommend(self, user_id:int):
        # preprocess
        ##
        if self.model_is_trained:
            if 

    def recommend_from_model1(self, user_id:int):
        return self._model_.recommend(user_id)

    def recommend_from_trending(self):
        pass

    def recommend_from_new(self):
        _len = len(self.product_list_count)
        return np.arange(_len, 0, -1, dtype=np.int)
    
    def recommend_at_random(self, array:np.ndarray):
        return np.random.shuffle(array)

    def recommend_from_model2(self):
        '''
        based on product time rather than rating.
        '''
            
