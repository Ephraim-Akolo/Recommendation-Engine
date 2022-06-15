import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from recommender import RecommendationSystem


app = FastAPI()

recommender = RecommendationSystem()


class UserData(BaseModel):
    user_id: int
    last_items: Union[list[int], None]

class NewUser(BaseModel):
    user_id: int

class NewItem(BaseModel):
    item_id: int

@app.get('/')
def index():
    return {"Server": "Running"}

@app.post("/recommend")
async def make_recommendation(userid:UserData):
    '''
    Make recommendations using the users unique serial number.
    '''
    recomm = None
    if userid.last_items:
        if len(userid.last_items) > 0:
            recomm = recommender.recommend(userid.user_id, userid.last_items)
        else:
            print("Error: Empty list")
    else:
        recomm = recommender.recommend(userid.user_id)
    return {"recommendation": recomm}

@app.post("/update_users")
def update_users(newuser:NewUser):
    '''
    Upates the recommenders database of users on user registration.
    '''
    return {"Success": recommender.update_users(newuser.user_id)}
    

@app.post("/update_items")
def update_items(newitem:NewItem):
    '''
    Updates the recommenders database of products(items) on product registration.
    '''
    return {"Success": recommender.update_items(newitem.item_id)}


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
    #uvicorn app:app --reload