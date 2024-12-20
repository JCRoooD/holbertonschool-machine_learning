-- Query the number of fans of each band, and order the result by the number of fans in descending order.
SELECT origin, SUM(fans) AS nb_fans
FROM metal_bands
GROUP BY origin
ORDER BY nb_fans DESC;
