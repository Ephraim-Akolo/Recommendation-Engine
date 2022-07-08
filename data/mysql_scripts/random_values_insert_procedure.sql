DELIMITER $$
CREATE PROCEDURE InsertRand(IN NumRows INT, IN MinVal INT, IN MaxVal INT)
    BEGIN
        DECLARE i INT;
        SET i = 1;
        START TRANSACTION;
        WHILE i <= NumRows DO
            INSERT INTO item_features(i1, i2, i3, i4, i5, i6, i7, i8, i9, i10) VALUES (
            RAND()*(MaxVal-MinVal)+MinVal,
            RAND()*(MaxVal-MinVal)+MinVal,
            RAND()*(MaxVal-MinVal)+MinVal,
            RAND()*(MaxVal-MinVal)+MinVal,
            RAND()*(MaxVal-MinVal)+MinVal,
            RAND()*(MaxVal-MinVal)+MinVal,
            RAND()*(MaxVal-MinVal)+MinVal,
            RAND()*(MaxVal-MinVal)+MinVal,
            RAND()*(MaxVal-MinVal)+MinVal,
            RAND()*(MaxVal-MinVal)+MinVal
            );
            SET i = i + 1;
        END WHILE;
        COMMIT;
    END$$
DELIMITER ;