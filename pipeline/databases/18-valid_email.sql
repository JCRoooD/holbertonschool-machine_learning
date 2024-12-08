-- This script creates a trigger that resets the valid_email field to 0 when the email field is changed.
DELIMITER //

CREATE TRIGGER reset_valid_email_on_change
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF OLD.email != NEW.email THEN
        SET NEW.valid_email = 0;
    END IF;
END //

DELIMITER ;