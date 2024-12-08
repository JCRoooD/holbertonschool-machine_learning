-- This script counts the number of shows for each genre.
SELECT tg.name AS genre, COUNT(tsg.genre_id) AS number_of_shows
FROM tv_genres tg, tv_show_genres tsg
WHERE tg.id = tsg.genre_id
GROUP BY genre
ORDER BY number_of_shows DESC;
