DROP PROCEDURE IF EXISTS InsertOnes;
DELIMITER $$
CREATE PROCEDURE InsertOnes(IN NumRows INT)
    BEGIN
        DECLARE i INT;
        SET i = 1;
        START TRANSACTION;
        WHILE i <= NumRows DO
            INSERT INTO bayesian_features(a, b, viewCount, clickedCount, added2cartCount, boughtCount) VALUES (1, 1, 0, 0, 0, 0);
            SET i = i + 1;
        END WHILE;
        COMMIT;
    END$$
DELIMITER ;