-- This script creates a function that safely divides two numbers.
DELIMITER //

DROP FUNCTION IF EXISTS SafeDiv //

CREATE FUNCTION SafeDiv(a INT, b INT) RETURNS FLOAT
DETERMINISTIC
BEGIN
    IF b = 0 THEN
        RETURN 0;
    ELSE
        RETURN a / b;
    END IF;
END //

DELIMITER ;