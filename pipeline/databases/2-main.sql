-- SQL script for 20-main.sql
SELECT * FROM users;
SELECT * FROM corrections;

SELECT "--";
CALL ComputeAverageScoreForUser((SELECT id FROM users WHERE name = "Jeanne"));

SELECT "--";
SELECT * FROM users;