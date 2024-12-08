-- This script creates a view that shows the rating of each TV show.
SELECT tv_shows.title, SUM(tv_show_ratings.rate) AS rating
FROM tv_show_ratings
JOIN tv_shows
ON tv_show_ratings.show_id = tv_shows.id
GROUP BY tv_shows.title
ORDER BY rating DESC;
