-- This script calculates the average temperature for each city and orders the results by the average temperature in descending order.
SELECT city, AVG(value) AS avg_temp FROM temperatures GROUP BY city ORDER BY avg_temp DESC;
