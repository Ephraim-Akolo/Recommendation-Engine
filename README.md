# SAKO RECOMMENDATION ENGINE PROJECT

Project statement: Build a recommendation engine for the Sako user application that recommends products from a database of products (groceries) in their categrories. 

### Milestones

    • Recommend to few users on new platform with little data (cold-start problem)
    • Recommend to users on growing platform
    • Recommend to users on established platform (with enough data to be fully functional)
    • Recommend products to new users after model is trained
    • Recommend new products to users after model is trained
    • online incremental training of model (warm training)
    • Retraining model occasionally to revive model performance (performance reduces with time)
    • Model deployment
    • Model serving

### Project targets

    • Model should be able to give useful recommendation custom to users.
    • Model should be able to retrain its self at post-determined specifications.
    • Model should be efficient and scalable.
    • Model should solve the cold-start problem.
    • Automated incremental learning

### Project features (data needed)

    • users serialized unique ID.
    • products serialized unique ID.
    • users ratings related to product.
    • users time spent on products.
    • users number of products/category/vendor visits.
    • (product description to train model on similar products)

## Recommendation System:

### Model
Model algorithm is currentlyy designed based on the bayesian beta distribution sampling. It is the model chosen to solve the current cold start problem of the application. The model has the advantage of exploiting and exploring at the same time. The algorithm is best for the current use case as it does not rely on any personalized data (we currently do not have).

__Other research algorithms:__

    • Matrix factorization
    • RNN/NLP
    • Clustering
    • Autoencoders
    • Restricted Boltzmann Machines

__Build language:__

    • Python
    • SQL
    • (C)

__Project Frameworks:__

    • FastAPI
    • MySQL
    • Pytorch
    • TensorFlow

__Project source code:__


[https://github.com/jake-ephraim/Recommendation-Engine/tree/master](https://github.com/jake-ephraim/Recommendation-Engine/tree/master)


## Recommendation system


## Deployment
Model will be deployed as a service i.e it will be hosted on a server where it can be ascessed through a post or get request. The model will be continously monitored and maintained by the machine learning team. This is because models intelligence tend to degrade overtime as human behavoir also changes with time, therefore, the model has to be retrained occationally and redeployed to service.


