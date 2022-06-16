from pathlib import Path
import numpy as np
from numpy.linalg import norm
##
from database import RecommenderDatabase


class RecommendationSystem():
    '''
    '''

    database = None

    _newest_frac = 0.1

    _trending_frac = 0.4

    def __init__(self, path:str=None, name:int=None) -> None:
        self.database = RecommenderDatabase(path, name)
        # if self.database.get_features_count():
        #     self.model_is_trained = True
    
    def is_viewtime_trained(self) -> bool:
        '''
        Returns True if the view time have been trained and uploaded to the database
        '''

    def is_rating_trained(self) -> bool:
        '''
        Returns True if ratings have been trained and uploaded to the database
        '''
    
    def is_new_user(user_id:int) -> bool:
        '''
        Returns True if the user features has not been pretrained (during the cold training)
        '''
    
    def merge_recommendation(rating_recommendations:np.ndarray, view_recommendations:np.ndarray) -> np.ndarray:
        '''
        Returns an vector of recommendations from the merged vectors of rating and view time recommendations.
        '''
    
    def promote_newest(recommendations:np.ndarray, trending_frac:float) -> np.ndarray:
        '''
        Increases the recommendation score (hierachy) of products based on newest score of the products.
        '''
    
    def recommend_from_last_viewed_items(self, last_viewed_items:list) -> np.ndarray:
        '''
        Finds similar items to last viewed item and mearged them into a single vector of recommendation.
        '''
    
    def cosine_similarity(self, product:np.ndarray, products:np.ndarray, decimal_place=2) -> float:
        '''
        Finds the cosine similarity between a vector and an array of vectors
        :param
        '''
        return np.dot(products,product)/(norm(products, axis=1)*norm(product))

    def recommend_trending(self):
        pass

    def recommend_newest(self):
        return

    def recommend_from_rating(self, user_id:int):
        return #self._model_.recommend(user_id)

    def recommend_from_viewtime(self, user_id:int):
        '''
        based on product time rather than rating.
        '''

    def recommend(self, user_id:int, last_veiwed_items:list=None):
        '''
        Make personalised user recommendation.
        '''
        is_rating_trained = self.is_rating_trained()
        is_viewtime_trained = self.is_viewtime_trained()
        recommendations = np.array([])
        if is_rating_trained and is_viewtime_trained:
            if self.is_new_user(user_id):
                if last_veiwed_items:
                    recommendations = self.recommend_from_last_viewed_items(last_veiwed_items)
                else:
                    recommendations = self.recommend_trending()
            else:
                rating_recommendations = self.recommend_from_rating()
                view_recommendations = self.recommend_from_viewtime()
                recommendations = self.merge_recommendation(rating_recommendations, view_recommendations)
                recommendations = self.promote_newest(recommendations, self._newest_frac)
        elif is_rating_trained:
            if self.is_new_user(user_id):
                if last_veiwed_items:
                    recommendations = self.recommend_from_last_viewed_items(last_veiwed_items)
                else:
                    recommendations = self.recommend_trending()
            else:
                recommendations = self.recommend_from_rating()
                recommendations = self.promote_newest(recommendations, self._newest_frac)
        elif is_viewtime_trained:
            if self.is_new_user(user_id):
                if last_veiwed_items:
                    recommendations = self.recommend_from_last_viewed_items(last_veiwed_items)
                else:
                    recommendations = self.recommend_trending()
            else:
                recommendations = self.recommend_from_viewtime()
                recommendations = self.promote_newest(recommendations, self._newest_frac)
        else:
            recommendations= self.recommend_newest()
        return recommendations.tolist()
    
    def update_users(self, user_id) -> bool:
        '''
        Updates the database when a new user registers.
        '''
        return False
    
    def update_items(self, item_id) -> bool:
        '''
        Updates the database when a new product is registered.
        '''
        return False

    # def __recommend__(self, user_id:int, last_veiwed_items:list=None):
    #     '''
    #     Make personalised user recommendation.
    #     '''
    #     # preprocess
    #     ##
    #     is_rating_trained = self.is_rating_trained()
    #     is_viewtime_trained = self.is_viewtime_trained()
    #     if is_rating_trained and is_viewtime_trained:
    #         if self.is_new_user(user_id):
    #             _rating_is_warmtrained = self.rating_is_warmtrained() 
    #             _viewtime_is_warmtrained = self.viewtime_is_warmtrained()
    #             if _rating_is_warmtrained and _viewtime_is_warmtrained:
    #                 rating_recommendations = self.recommend_from_rating()
    #                 view_recommendations = self.recommend_from_viewtime()
    #                 recommendations = self.merge_recommendation(rating_recommendations, view_recommendations)
    #                 recommendations = self.promote_trending(recommendations, self._trending_frac)
    #             elif _rating_is_warmtrained:
    #                 recommendations = self.recommend_from_rating()
    #                 recommendations = self.promote_trending(recommendations, self._trending_frac)
    #             elif _viewtime_is_warmtrained:
    #                 recommendations = self.recommend_from_viewtime()
    #                 recommendations = self.promote_trending(recommendations, self._trending_frac)
    #             else:
    #                 recommendations = self.recommend_trending()
    #         else:
    #             rating_recommendations = self.recommend_from_rating()
    #             view_recommendations = self.recommend_from_viewtime()
    #             recommendations = self.merge_recommendation(rating_recommendations, view_recommendations)
    #             recommendations = self.promote_newest(recommendations, self._newest_frac)
    #     elif is_rating_trained:
    #         if self.is_new_user(user_id):
    #             if self.rating_is_warmtrained():
    #                 recommendations = self.recommend_from_rating()
    #                 recommendations = self.promote_trending(recommendations, self._trending_frac)
    #             else:
    #                 recommendations = self.recommend_trending()
    #         else:
    #             recommendations = self.recommend_from_rating()
    #             recommendations = self.promote_newest(recommendations, self._newest_frac)
    #     elif is_viewtime_trained:
    #         if self.is_new_user(user_id):
    #             if self.viewtime_is_warmtrained():
    #                 recommendations = self.recommend_from_viewtime()
    #                 recommendations = self.promote_trending(recommendations, self._trending_frac)
    #             else:
    #                 recommendations = self.recommend_trending()
    #         else:
    #             recommendations = self.recommend_from_viewtime()
    #             recommendations = self.promote_newest(recommendations, self._newest_frac)
    #     else:
    #         recommendations= self.recommend_newest()
    #     return recommendations

    # def rating_is_warmtrained(user_id:int) -> bool:
    #     '''
    #     Returns True if the user rating features has been warm trained by the warm training engine.
    #     '''
    
    # def viewtime_is_warmtrained(user_id:int) -> bool:
    #     '''
    #     Returns True if the user view time features has been warm trained by the warm training engine.
    #     '''
            
    # def promote_trending(recommendations:np.ndarray, trending_frac:float) -> np.ndarray:
    #     '''
    #     Increases the recommendation score (hierachy) of products based on trending score of the products.
    #     '''