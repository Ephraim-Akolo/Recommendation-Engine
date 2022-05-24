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
            cur.execute("CREATE TABLE IF NOT EXISTS utility(num_features INT, num_trained_user INT, num_trained_product INT, trained INT)")
            cur.execute("INSERT INTO utility (num_features, num_trained_user, num_trained_product, trained) VALUES (2, 0, 0, 0)")
            self.conn.commit()
        except sqlite3.Error:
            if self.conn:
                print("Error! Rolling back changes")
                self.conn.rollback()
    
    def make_features(self, num:int=1):
        '''
        '''
        cur = self.conn.cursor()
        col_count = self.get_count()
        if num <= col_count:
            return
        for i in range(col_count+1, num+1):
            cur.execute(f"ALTER TABLE user_features ADD feature_{i} REAL")
            cur.execute(f"ALTER TABLE product_features ADD feature_{i} REAL")
        cur.execute("UPDATE utility SET num_features=? where rowid=1", (num,))
        self.conn.commit()
    
    def get_count(self, feature=True):
        '''
        '''
        cur = self.conn.cursor()
        if feature:
            return cur.execute("SELECT num_features FROM utility LIMIT 1").fetchall()[0][0]
        else:
            return cur.execute("SELECT num_trained_user, num_trained_product FROM utility LIMIT 1").fetchall()[0]
    
    def get_product_features(self):
        cur = self.conn.cursor()
        return cur.execute("SELECT * FROM product_features ORDER BY rowid ASC").fetchall()
    
    def get_user_feature(self, user_id):
        cur = self.conn.cursor()
        return cur.execute("SELECT * FROM user_features WHERE rowid = ?", (user_id,)).fetchall()

    def update_database(self, array:list, table:str="user_features"):
        row_count = len(array)
        col_count = len(array[0])
        trained_count = self.get_count(False)
        self.make_features(col_count)
        cur = self.conn.cursor()
        s1 = self.generate_features_name(col_count)
        if row_count > trained_count[0] and table == "user_features":
            cur.execute("UPDATE utility SET num_trained_user=? where rowid=1", (row_count,))
            s2 = self.generate_features_name(col_count, True)
        elif row_count > trained_count[1] and table == "product_features":
            cur.execute("UPDATE utility SET num_trained_product=? where rowid=1", (row_count,))
            s2 = self.generate_features_name(col_count, True)
        for rowid, l in enumerate(array):
            if (rowid < trained_count[0] and table == "user_features") or (rowid < trained_count[1] and table == "product_features"):
                cur.execute(f'UPDATE {table} SET {s1} where rowid = ?', (*l, rowid+1))
            else:
                cur.execute(f'INSERT INTO {table} ({s2[0]}) VALUES ({s2[1]}) ', l)
        self.conn.commit()
        return True
    
    def generate_features_name(self, count:int, sep=False):
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
        inc = len(array)
        f = self.generate_features_name(len(array[0]), sep=True)
        for l in array:
            cur.execute(f"INSERT INTO {table} ({f[0]}) VALUES ({f[1]})", l)
        if table == "user_features":
            col = f"num_trained_user = num_trained_user+{inc}" 
        elif table == "product_features":
            col = f"num_trained_product = num_trained_product+{inc}"
        cur.execute(f'UPDATE utility SET {col} where rowid=1')
        self.conn.commit()
        return True


