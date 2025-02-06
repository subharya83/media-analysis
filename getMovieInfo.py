import os
import re
import argparse
import pandas as pd
import imdb
import logging

class IMDBMetaData:
    def __init__(self):
        self.conn = imdb.IMDb()

    def _extract_year_from_title(self, title):
        """Extract the year from the movie title using regex."""
        year_match = re.search('\([1-9]\d\d\d\)', title)
        return int(year_match.group(0).replace('(', '').replace(')', '')) if year_match else None

    def _build_query(self, title, year):
        """Build a query string for IMDb search."""
        return f"{title} ({year})" if year else title

    def _search_movie(self, query):
        """Search for a movie on IMDb using the given query."""
        return self.conn.search_movie(query)

    def get_imdb_info_from_file(self, file_path):
        """Get IMDb information for movies listed in a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        df = pd.read_csv(file_path, sep="\t")
        titles = df['MovieTitle']
        years = [self._extract_year_from_title(title) for title in df.iloc[:, -1]]

        results = []
        for title, year in zip(titles, years):
            query = self._build_query(title, year)
            movies = self._search_movie(query)
            movie_id = movies[0].movieID if movies else 'IMDB_INFO_NA'
            results.append((year, title, movie_id))
            print(year, title, movie_id)

        return results

    def get_imdb_info_by_title_year(self, title, year=None):
        """Get IMDb information for a specific movie title and year."""
        query = self._build_query(title, year)
        movies = self._search_movie(query)
        movie_id = movies[0].movieID if movies else 'IMDB_INFO_NA'
        return year, title, movie_id

def main():
    parser = argparse.ArgumentParser(description='Download IMDb info given title and year')
    parser.add_argument('--file', '-f', type=str, help='Path to metadata index file')
    parser.add_argument('--title', '-t', type=str, help='Movie Title')
    parser.add_argument('--year', '-y', type=str, help='Movie Year')

    args = parser.parse_args()

    db = IMDBMetaData()

    if args.file:
        db.get_imdb_info_from_file(args.file)
    elif args.title:
        result = db.get_imdb_info_by_title_year(args.title, args.year)
        print(result)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()