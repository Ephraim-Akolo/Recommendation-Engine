from multiprocessing import shared_memory
import numpy as np
from numpy.linalg import norm
import json
# from optimized.optimized_functions import merge_recommendations_
##
from recommendationEngines.database.database import RecommenderDatabase
from os import environ
import logging
logger =logging.getLogger(environ.get("SAKO_LOGGER_NAME"))


class RecommendationBase():
    '''
    '''

    database = None

    _newest_frac = 0.1

    _trending_frac = 0.4

    MAX_RECOMM_SIZE = 1000

    shared_mem_name = None

    shared_mem_shape = None

    shared_mem_datatype = None

    product_list = None

    def __init__(self) -> None:
        self.database = RecommenderDatabase()
        self.last_user_id = None
        self.refresh_config()
        self.refresh_data_from_db()
    
    def refresh_data_from_db(self) -> bool:
        '''
        Updates and cache variables from the database.
        '''
        self.last_user_id = self.database.get_last_user_id()
        #
        return True

    def refresh_config(self) -> bool:
        '''
        Refreshes the configurations from the config file.
        '''
        try:
            with open("./config.json", 'r') as fp:
                self.config = json.load(fp)
            return True
        except Exception as e:
            self.config = None
            logger.error(f"FAILED TO READ CONFIG FILE WITH ERROR: {e}")
            return False    

    def update_config(self, data:dict):
        '''
        Updates the configuration file and object with the new values from data.
        '''
        updated_config = []
        try:
            if not (len(data) > len(self.config)):
                for key, val in data.items():
                    if key in self.config.keys() and type(val) is type(self.config[key]):
                        self.config[key] = val
                        updated_config.append(key)
                if len(updated_config) > 0:
                    with open("./config.json", 'w') as fp:
                        json.dump(self.config, fp)
            return updated_config
        except Exception as e:
            logger.error(f"FAILED TO UPDATE CONFIG FILE WITH ERROR: {e}")
            return None

    def is_trained(self) -> tuple[bool, bool]:
        '''
        Returns True if ratings or view time have been trained and uploaded to the database
        '''
        if self.config:
            pass
            # return self.config['is_rating_trained'], self.config['is_viewtime_trained']
        else:
            # return False, False
            pass
        return False, False
    
    def is_new_user(self, user_id:int) -> bool:
        '''
        Returns True if the user features has not been pretrained (during the cold training)
        '''
        if self.last_user_id > 0 and user_id <= self.last_user_id:
            #return False
            pass
        return True

    def merge_recommendation3(self, similarities_1:np.ndarray, similarities_2:np.ndarray, similarities_3:np.ndarray) -> list:
        '''
        Returns an vector of recommendations from the merged vectors of rating and view time recommendations.
        '''
        max_count = self.MAX_RECOMM_SIZE
        cs_1 = similarities_1.argsort()[::-1][1:max_count+1].astype(np.uint32)
        cs_2 = similarities_2.argsort()[::-1][1:max_count+1].astype(np.uint32)
        cs_3 = similarities_3.argsort()[::-1][1:max_count+1].astype(np.uint32)
        recommended = bytearray(self.shared_mem_shape[0])
        recommendations = []
        n = 0
        _zip = zip(cs_1, cs_2, cs_3)
        next(_zip)
        for i, j, k in _zip:
            if not recommended[i]:
                recommendations.append(i+1)
                recommended[i] = 1
                n += 1
            if not recommended[j]:
                recommendations.append(j+1)
                recommended[j] = 1
                n += 1
            if not recommended[k]:
                recommendations.append(k+1)
                recommended[k] = 1
                n += 1
            raise(Exception("Not completely implemented!"))
    
    def merge_recommendation(self, similarities_1:np.ndarray, similarities_2:np.ndarray) -> list:
        '''
        Returns an vector of recommendations from the merged vectors of rating and view time recommendations.
        '''
        max_count = self.MAX_RECOMM_SIZE
        cs_1 = similarities_1.argsort()[::-1][1:max_count+1].astype(np.uint32)
        cs_2 = similarities_2.argsort()[::-1][1:max_count+1].astype(np.uint32)
        # return merge_recommendations_(cs_1, cs_2, self.shared_mem_shape[0], max_count).tolist()
        recommended = bytearray(self.shared_mem_shape[0])
        recommendations = []
        n = 0
        _zip = zip(cs_1, cs_2)
        next(_zip)
        for i, j in _zip:
            if not recommended[i]:
                recommendations.append(i+1)
                recommended[i] = 1
                n += 1
            if not recommended[j]:
                recommendations.append(j+1)
                recommended[j] = 1
                n += 1
            if not n < max_count:
                break
        return recommendations[:max_count]
            
    def recommend_from_last_viewed_items(self, last_viewed_items:list) -> list:
        '''
        Finds similar items to last viewed item and mearged them into a single vector of recommendation.
        '''
        item_features:np.ndarray = self.product_list[[i-1 for i in last_viewed_items]]
        last_item_features = item_features[-1]
        l = len(last_viewed_items)
        if l == 1:
            return (self.cosine_similarity(last_item_features).argsort()[::-1]+1)[1:self.MAX_RECOMM_SIZE+1].tolist()
        elif l <= 5:
            disimilar_item_features = item_features[self.cosine_similarity(last_item_features, item_features).argmin()]
            #return self.merge_recommendation(self.cosine_similarity(last_item_features), self.cosine_similarity(disimilar_item_features))
            
            r = self.merge_recommendation(self.cosine_similarity(last_item_features), self.cosine_similarity(disimilar_item_features))
        elif l < 10:
            disimilar_item_features = item_features[self.cosine_similarity(last_item_features, item_features).argsort()[l//2]]
            #return self.merge_recommendation(self.cosine_similarity(last_item_features), self.cosine_similarity(disimilar_item_features))
            
            r = self.merge_recommendation(self.cosine_similarity(last_item_features), self.cosine_similarity(disimilar_item_features))
        else:
            _sorted = self.cosine_similarity(last_item_features, item_features).argsort()
            disimilar_item_features = item_features[_sorted[int(l-l*0.7)]]
            midsimilar_item_features = item_features[_sorted[int(l-l*0.5)]]
            #return self.merge_recommendation(self.cosine_similarity(last_item_features), self.cosine_similarity(disimilar_item_features))
            
            r = self.merge_recommendation(self.cosine_similarity(last_item_features), self.cosine_similarity(midsimilar_item_features), self.cosine_similarity(disimilar_item_features))
        raise(Exception("Not completely implemented!"))

    def cosine_similarity(self, product:np.ndarray, products:np.ndarray = None, decimal_place=2) -> np.ndarray:
        '''
        Finds the cosine similarity between a vector and an array of vectors
        :param
        product: a vector
        products: an array of vectors (2D)
        '''
        if not isinstance(products, np.ndarray):
            products = self.product_list
        return np.dot(products,product)/(norm(products, axis=1)*norm(product))

    def recommend_trending(self):
        raise(NotImplementedError('"recommend_trending" method not implemented!'))

    def recommend_from_rating(self, user_id:int):
        raise(NotImplementedError('"recommend_from_rating" method not implemented!'))

    def recommend_from_viewtime(self, user_id:int):
        '''
        based on product time rather than rating.
        '''
        raise(NotImplementedError('"recommend_from_viewtime" method not implemented!'))

    def recommend(self, user_id:int, last_veiwed_items:list=None):
        '''
        Make personalised user recommendation.
        '''
        is_rating_trained, is_viewtime_trained = self.is_trained()
        recommendations = None
        if is_rating_trained and is_viewtime_trained:
            if self.is_new_user(user_id):
                if last_veiwed_items:
                    recommendations = self.recommend_from_last_viewed_items(user_id, last_veiwed_items)
                else:
                    recommendations = self.recommend_trending()
            else:
                rating_recommendations = self.recommend_from_rating()
                view_recommendations = self.recommend_from_viewtime()
                recommendations = self.merge_recommendation(rating_recommendations, view_recommendations)
                #recommendations = self.promote_newest(recommendations, self._newest_frac)
        elif is_rating_trained:
            if self.is_new_user(user_id):
                if last_veiwed_items:
                    recommendations = self.recommend_from_last_viewed_items(last_veiwed_items)
                else:
                    recommendations = self.recommend_trending()
            else:
                recommendations = self.recommend_from_rating()
                #recommendations = self.promote_newest(recommendations, self._newest_frac)
        elif is_viewtime_trained:
            if self.is_new_user(user_id):
                if last_veiwed_items:
                    recommendations = self.recommend_from_last_viewed_items(last_veiwed_items)
                else:
                    recommendations = self.recommend_trending()
            else:
                recommendations = self.recommend_from_viewtime()
                #recommendations = self.promote_newest(recommendations, self._newest_frac)
        else:
            recommendations =  self.recommend_newest()
        assert type(recommendations) == list
        raise(NotImplementedError('"recommender" method not Re-implemented!'))
        return recommendations
    
    def update_items(self, item_id) -> bool:
        '''
        Updates the database when a new product is registered.
        '''
        return False
    
    def load_products_to_memory(self, name:str='shared_product_memory'):
        products_array = self.database.get_all_products()
        self.shared_mem_name = name
        self.shared_mem_shape = products_array.shape
        self.shared_mem_datatype = np.float32
        d_size = np.dtype(self.shared_mem_datatype).itemsize * np.prod(self.shared_mem_shape)
        n = 0
        while 1:
            try:
                shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
                break
            except FileExistsError as e:
                n += 1
                logger.info("Renaming memory!")
                self.shared_mem_name = name = name + str(n)
            except Exception as e:
                raise(e)
        # numpy array on shared memory buffer
        product_list = np.ndarray(shape=self.shared_mem_shape, dtype=self.shared_mem_datatype, buffer=shm.buf)
        product_list[:] = products_array[:]
        return True
    
    def attach_to_products_memory(self):
        self.shm = shared_memory.SharedMemory(name=self.shared_mem_name)
        # numpy array on shared memory buffer
        self.product_list = np.ndarray(shape=self.shared_mem_shape, dtype=self.shared_mem_datatype, buffer=self.shm.buf)

    def detach_from_products_memory(self):
        self.shm.close()
    
    def release_products_memory(self):
        shm = shared_memory.SharedMemory(name=self.shared_mem_name)
        shm.close()
        shm.unlink()  # Free and release the shared memory block

    def __call__(self,  user_id:int, last_veiwed_items:list=None) -> list:
        self.attach_to_products_memory()
        recommendations = self.recommend(user_id, last_veiwed_items)
        self.detach_from_products_memory()
        return recommendations

    
if __name__ == "__main__":
    pass