import sqlite3
from pathlib import Path


class RecommenderDatabase:

    database_name = "sako_db.db"

    database_path = Path("./data/databases/")

    conn = None
    
    def __init__(self, path:str=None, name:str=None) -> None:
        if path:
            self.database_path = Path(path)
        if name:
            self.database_name = name
        self.create_database()

    def create_database(self, path:str=None, name:str=None, R:int=2):
        """creates a database in the specified path
        """
        if path:
            self.database_path = Path(path)
        if name:
            self.database_name = name
        try:
            self.conn = sqlite3.connect(str(self.database_path/self.database_name))
            cur = self.conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS user_features(feature_1 REAL, feature_2 REAL)")
            cur.execute("CREATE TABLE IF NOT EXISTS product_features(feature_1 REAL, feature_2 REAL)")
            cur.execute("CREATE TABLE IF NOT EXISTS utility (num_features INT, num_trained INT, trained INT)")
            self.conn.commit()
        except sqlite3.Error:
            if self.conn:
                print("Error! Rolling back changes")
                self.conn.rollback()
    
    def make_features(self, num:int=1):
        '''
        '''
        cur = self.conn.cursor()
        columns_num = self.get_features_count()
        if columns_num > num:
            return
        for i in range(columns_num+1, num+1):
            cur.execute(f"ALTER TABLE user_features ADD feature_{i} REAL")
            cur.execute(f"ALTER TABLE product_features ADD feature_{i} REAL")
        self.conn.commit()
    
    def get_features_count(self):
        cur = self.conn.cursor()
        return len(cur.execute("SELECT * FROM product_features LIMIT 1").fetchall()[0])
    
    def get_product_features(self):
        cur = self.conn.cursor()
        return cur.execute("SELECT * FROM product_features ORDER BY rowid ASC").fetchall()
    
    def get_user_feature(self, user_id):
        cur = self.conn.cursor()
        return cur.execute("SELECT * FROM user_features WHERE rowid = ?", (user_id,)).fetchall()

    def update_database(self, array:list, table:str="user_features"):
        count = len(array[0])
        if count != self.get_features_count():
            print("Error")
            return False
        cur = self.conn.cursor()
        s = self.generate_features(count)
        comm = f'UPDATE {table} SET {s} where rowid = ?'
        for rowid, l in enumerate(array):
            cur.execute(comm, (*l, rowid+1))
        self.conn.commit()
        return True
    
    def generate_features(self, count:int, sep=False):
        if sep:
            s = ["feature_1", "?"]
            for i in range(count):
                if i+1 == 1:
                    continue
                s[0] = s[0] + (f",feature_{i+1}")
                s[1] = s[1] + (",?")
        else:
            s = "feature_1 = ?"
            for i in range(count):
                if i+1 == 1:
                    continue
                s = s + f",feature_{i+1} = ?"
        return s
    
    def insert_into_table(self, array:list, table:str='user_features'):
        cur = self.conn.cursor()
        f = self.generate_features(len(array[0]), sep=True)
        for l in array:
            cur.execute(f"INSERT INTO {table} ({f[0]}) VALUES ({f[1]})", l)
        self.conn.commit()
        return True


