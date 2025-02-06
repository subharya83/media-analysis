import os
import re
import argparse
import pandas as pd
import imdb
import logging


class IMDBMetaData:
    def __init__(self):
        self.conn = imdb.IMDb();

    def getIMDB_info(self, _path=None):
        if os.path.exists(_path):
            df = pd.read_csv(_path, sep="\t")
            # Assuming last column is the YouTube video Title
            ttl = df.iloc[:, -1]
            # Getting a list of years
            _years = [y.group(0) if y is not None else None for y in (re.search('\([1-9]\d\d\d\)', t) for t in ttl)]
            _years = [int(y.replace('(', '').replace(')', '')) if y is not None else None for y in _years]
            # Getting a list of movie titles
            _titles = [t for t in df['MovieTitle']]

            # Use this to create a query
            _idx = 0
            for _title in _titles:
                _year = _years[_idx]
                if _year:
                    _query = _title + ' (' + str(_year) + ')'
                else:
                    _query = _title
                m = self.conn.search_movie(_query)
                if len(m) > 0:
                    #logging.info(_year, _title, m[0].movieID)
                    print(_year, _title, m[0].movieID)
                else:
                    #logging.info(_year, _title, 'IMDB_INFO_NA')
                    print(_year, _title, 'IMDB_INFO_NA')
                _idx += 1

    def getIMDB_info_title_year(self, _title=None, _year=None):
        if not _title:
            return _year, _title, 'IMDB_INFO_NA'
        if _year:
            _query = _title + ' (' + str(_year) + ')'
        else:
            _query = _title
        m = self.conn.search_movie(_query)
        if len(m) > 0:
            #logging.info(_year, _title, m[0].movieID)
            return _year, _title, m[0].movieID
        else:
            #logging.info(_year, _title, 'IMDB_INFO_NA')
            return _year, _title, 'IMDB_INFO_NA'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, description='Download IMDB info given title and year')
    #parser.add_argument('--mpath', '-m', type=str, required=True, help='Path to metadata index file')
    parser.add_argument('--title', '-t', type=str, required=True, help='Movie Title')
    parser.add_argument('--year', '-y', type=str, required=True, help='Movie Year')

    args = parser.parse_args()

    db = IMDBMetaData()
    #db.getIMDB_info(args.mpath)
    db.getIMDB_info_title_year(_title=args.title, _year=args.year)