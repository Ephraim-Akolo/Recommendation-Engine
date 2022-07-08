import numpy as np
from recommendationEngines.base import RecommendationBase


class Recommender(RecommendationBase):
    
    def recommend(self, user_id: int, last_veiwed_items: list = None):
        '''
        Method called to return recommendations to the user.
            :param user_id: the unique serial number of the user.
            :param last_viewed_items: a list of unique serial numbers of the products (items) the user recently viewed.
            :returns: a list of recommendations for the user.
        '''
        return self.recommend_trending()
    
    def recommend_trending(self) -> list:
        '''
        Make recommendations based of sample datapoints from the bayesian
            :returns: a list of recommendations for the user sampled from the bayesian beta distribution.
        '''
        anb = self.database.get_beta_values()
        rng = np.random.default_rng()
        ranking = rng.beta(anb[:, 0], anb[:, 1])
        return np.argsort(ranking)[-self.MAX_RECOMM_SIZE:][::-1].tolist()


if __name__ == "__main__":
    print(Recommender())
    
