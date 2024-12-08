-- This script creates a table that contains the genre_id for each show in the database.
SELECT tv_shows.title, tv_show_genres.genre_id FROM tv_shows, tv_show_genres
WHERE tv_shows.id = tv_show_genres.show_id
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;
