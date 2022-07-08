CREATE DATABASE IF NOT EXISTS SakoDB;
USE SakoDB;
DROP TABLE IF EXISTS tables_data;
CREATE TABLE tables_data(
item_features_len INT NOT NULL,
view_product_features_len INT NOT NULL,
view_user_features_len INT NOT NULL,
rating_product_features_len INT NOT NULL,
rating_user_features_len INT NOT NULL
);
INSERT INTO tables_data values(
10,
0,
0,
0,
0
);