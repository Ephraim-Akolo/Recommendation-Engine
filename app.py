import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Union
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import asyncio
import json
from hashlib import sha256
import logging
from os import environ


app = FastAPI()

logger_name = "gunicorn.error" if __name__ != "__main__" else "uvicorn.error"
environ["SAKO_LOGGER_NAME"] = logger_name
logger = logging.getLogger(logger_name) 


class UserData(BaseModel):
    user_id: int
    last_items: Union[list[int], None] = None
    acces_code: str

class NewUser(BaseModel):
    user_id: int
    access_code: str

class NewItem(BaseModel):
    item_id: int
    access_code: str

class PassCode(BaseModel):
    password: str
    data: Union[dict, None] = None

class BayesianData(BaseModel):
    data: Dict[str,int]
    access_code: str


@app.get('/')
async def index():
    '''
    Application root.
    '''
    return {"Server": "Running", "engine": app.state.recommender.name_, "server_processors": cpu_count()}

async def recommend_in_process(fn, user_id:int, last_items:list):
    '''
    Sends each recommendation request to a new process.
        :param fn: recommendation object to be sent to a process.
        :param user_id: the unique serial number of the user to be passed as argument to fn.
        :param last_items: a list of unique serial numbers of items the user last viewed.
        :returns: the recommendation made from the process awaited.
    '''
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(app.state.executor, fn, user_id, last_items)

@app.post("/recommend")
async def make_recommendation(userid:UserData):
    '''
    Make recommendations using the users unique serial number.
        :param userid: the unique serial number of the user.
        :returns: a json (map) of the recommendations made.
    '''
    if hash_password(userid.acces_code) == app.state.ACCESS_CODE:
        recomm = None
        if userid.last_items is not None and len(userid.last_items) == 0:
            userid.last_items = None
        recomm = await recommend_in_process(app.state.recommender, userid.user_id, userid.last_items)
        return {"recommendations": recomm}
    return {"Unauthorized User"}

@app.post("/update_bayesian_features")
async def update_bayesian_db(data:BayesianData):
    '''
    update seen(0), click(1) and buy(3) status of the products (items) seen by the user.
        :param data: A dictionary mapping the interaction of a user and the products seen.
        :returns: True if successfully updated otherwise False.
    '''
    if hash_password(data.access_code) == app.state.ACCESS_CODE:
        if len(data.data) == 0:
            return False
        d0, d1, d2 = [], [], []
        for key, val in data.data.items():
            if val == 0:
                d0.append(int(key))
            elif val == 1:
                d1.append(int(key))
            elif val == 3:
                d2.append(int(key))
        l = []
        if len(d0) == 1:
            l.append(f'UPDATE bayesian_features SET b=b+1, viewCount=viewCount+1 WHERE id={d0[0]}')
        elif len(d0) > 1:
            l.append(f'UPDATE bayesian_features SET b=b+1, viewCount=viewCount+1 WHERE id IN {tuple(d0)}')
        if len(d1) == 1:
            l.append(f'UPDATE bayesian_features SET a=a+0.7, viewCount=viewCount+1, clickedCount=clickedCount+1 WHERE id={d1[0]}')
        elif len(d1) > 1:
            l.append(f'UPDATE bayesian_features SET a=a+0.7, viewCount=viewCount+1, clickedCount=clickedCount+1 WHERE id IN {tuple(d1)}')
        if len(d2) == 1:
            l.append(f'UPDATE bayesian_features SET a=a+1, viewCount=viewCount+1, clickedCount=clickedCount+1, boughtCount=boughtCount+1 WHERE id={d2[0]}')
        elif len(d2) > 1:
            l.append(f'UPDATE bayesian_features SET a=a+1, viewCount=viewCount+1, clickedCount=clickedCount+1, boughtCount=boughtCount+1 WHERE id IN {tuple(d2)}')
        if len(l) < 0:
            return False
        if app.state.recommender.database.update_bayesian_db(*l):
            return True
        return False
    return {"Unauthorized User"}

@app.post("/update_bayesian_items")
async def update_bayesian_items(newitem:NewItem):
    '''
    Add items to database on product registration.
        :param newitem: the unique serial number of the item enlisted.
        :returns: True if item is successfully added and False if something went wrong.
    '''
    if hash_password(newitem.access_code) == app.state.ACCESS_CODE:
        return app.state.recommender.database.update_bayesian_items(newitem.item_id)
    return {"Unauthorized User"}

@app.post("/refresh_data_from_db")
async def refresh_data_from_db(passcode:PassCode):
    '''
    Updates and cache variables from the database. "data" is null.
        :param name: the apps password. The data is negleted.
        :returns: the True if successful otherwise False
    '''
    if hash_password(passcode.password) == app.state.PASS_CODE:
        return app.state.recommender.refresh_data_from_db()
    return {"Unauthorized User"}

@app.post("/reload_model")
async def reload_model(name:PassCode):
    '''
    Reload the recommendation engine being used to the engine specified.
        :param name: the apps password and name of the Recommender to load.
        :returns: True if successful and None if it fails.
    '''
    if hash_password(name.password) == app.state.PASS_CODE:
        recommender = load_model(name.data[tuple(name.data.keys())[0]])
        recommender.shared_mem_name = app.state.recommender.shared_mem_name
        recommender.shared_mem_shape = app.state.recommender.shared_mem_shape
        recommender.shared_mem_datatype = app.state.recommender.shared_mem_datatype
        app.state.recommender = recommender
        # update config
        return True
    return {"Unauthorized User"}

def load_model(name:str=None):
    '''
    Function that imports and load the recommendation model.
        :param name: the file name of the Recommendation model.
        :returns: the the model if successful and None if the model importation fail's.
    '''
    if name:
        engine_name = name
        app.state.recommender.update_config({"engine_name": name})
    else:
        try:
            with open("./config.json", 'r') as fp:
                d = json.load(fp)
                engine_name = d['engine_name']
        except:
            engine_name = "recommendationEngine_v1_0"
    try:
        exec(f"from recommendationEngines.{engine_name} import Recommender", globals())
        recommender = Recommender()
        recommender.name_ = engine_name
        logger.info(f'{engine_name} loaded!')
        return recommender
    except Exception as e:
        logger.error(e)
        return None
    
def hash_password(password:str):
    '''
    A cryptographic hash function that creates the hast of a keyword or password.
        :param password: the keyword to be hashed.
        :returns: a hex string.
    '''
    return sha256(password.encode("utf-8")).hexdigest()

def create_config(config_path = "./config.json") -> bool:
        '''
        Creates a new configuration file if it does not already exist in directory.
            :param config_path: the path of the config file to create.
            :return: returns True if successful and False if the creation fails.
        '''
        data = {
            'engine_name': 'recommendationEngine_v1_0'
        }
        try:
            with open(config_path, 'r'):
                pass
            logger.info("CANNOT CREATE CONFIG FILE!!! FILE EXIST.")
            print("CANNOT CREATE CONFIG FILE!!! FILE EXIST.")
            return False
        except FileNotFoundError:
            try:
                with open('./config.json', 'w') as fp:
                    json.dump(data, fp)
                return True
            except Exception as e:
                logger.error("FAILED TO CREATE THE CONFIG FILE WITH ERROR: {e}")
                print("FAILED TO CREATE THE CONFIG FILE WITH ERROR: ", e)
                return False
        except Exception as e:
            logger.error(f"UNKNOWN ERROR: {e}")
            print("UNKNOWN ERROR: ", e)

@app.on_event("startup")
async def on_startup():
    app.state.executor = ProcessPoolExecutor(max_workers=cpu_count())
    create_config()
    app.state.recommender = load_model()
    app.state.recommender.load_products_to_memory()
    app.state.PASS_CODE = '8f07077afffb95cb192fe250f7bd10fc47a9a263ff69d6ccbc2889eeee43242b'
    app.state.ACCESS_CODE = 'dabd42bd0284c2d7f3038df67539cba5b5c0317c8b969f96b5e6643f333c7308'

@app.on_event("shutdown")
async def on_shutdown():
    app.state.executor.shutdown()
    app.state.recommender.release_products_memory()

if __name__ == "__main__":
    pass
    #uvicorn.run(app, host='127.0.0.1', port=8004)
    #uvicorn app:app --reload