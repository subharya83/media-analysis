# Movie Scenes,Trailer and Script Analyzer (MovieSTAr Dataset)

## Dataset Creation 

### Collecting Videos
1. Downloading videos IDs from YouTube for a given channel id:
```
youtube-dl -i --get-id https://www.youtube.com/user/MovieclipsTrailers | tee MovieTrailers.txt
youtube-dl -i --get-id https://www.youtube.com/user/Movieclips | tee MovieScenes.txt
```
2. Downloading videos IDs uploaded after a particular date (YYYYMMDD):
#### Trailers
```
youtube-dl -i --dateafter 20191201 --match-filter "like_count > 100 & dislike_count <? 50" --get-id --get-title --get-duration --match-title "trailer"  https://www.youtube.com/user/MovieclipsTrailers | tee MovieTrailers.txt
youtube-dl -i --dateafter 20191201 --match-filter "like_count > 100 & dislike_count <? 50" --get-id --get-title --get-duration --match-title "trailer"  https://www.youtube.com/user/Filme | tee MovieTrailers-2.txt 
```
#### Scenes
```
youtube-dl -i --dateafter 20191201 --match-filter "like_count > 100 & dislike_count <? 50" --get-id --get-title --get-duration https://www.youtube.com/user/MovieClips | tee MovieScenes.txt
```
3. Getting videos from YouTube according to a file containing YouTube Ids
```
youtube-dl -i --output "YTID_%(id)s.%(ext)s" -a <file.txt> -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'
```
4. Converting videos in other formats to mp4
```
for i in `ls *.webm`; do x=`echo $i|sed 's/\.webm/\.mp4/g'`; ffmpeg -y -i $i -crf 5 -strict -2 $x; done
```
### Scripts
```shell
# Download movie scripts

# Canonicalize html to text 
for i in $(find ../data/movie-scripts/ -name \*.htm); do 
	x=$(echo $i|sed 's/\.htm/\.txt/g');
	python3 html2text.py $i>$x;
done
```
### Getting YouTube video title 
```
for i in `ls stats/*.json`; do 
  id=`basename $i|sed -e 's/\.json//g' -e 's/TTD_//g'`; 
  ttl=`jq .Title $i`; 
  echo $id,$ttl; 
done > >(tee metadata/MovieTitles.csv) 
```

### Extracting Year of Movie from YouTube video title string
#### Using Shell
```
f=metadata/titles.tsv;
nlines=`cat $f|wc -l`; 
for i in `seq 1 $nlines`; do 
  ln=`awk "NR==$i" $f`; 
  year=`echo $ln|sed -n 's/.*\([1-9][0-9][0-9][0-9]\).*/\1/p'`; 
  echo $year;  
done > >(tee metadata/MovieTitles_Year.csv)
```
#### Using python
```
import pandas as pd
import re
df = pd.read_csv('MTMI.tsv', sep='\t')
ttl = df.iloc[:, -1]
_years = [y.group(0) if y is not None else None for y in (re.search('\([1-9][0-9][0-9][0-9]\)', t) for t in ttl)]
_years = [int(y.replace('(', '').replace(')', '')) if y is not None else None for y in _years]
```

### Clean up VideoId Title field to extract movie title from Trailer video titles
```
#!/bin/bash
for x in `seq 0 5202`; 
  do x=`jq .Title.\"$x\" MTMI.json|sed -e 's/\(Official\|International\|Final\|Comic-Con\|Teaser\|Red Band\|Movie\).* Trailer.*$//g'`;
  echo $x; 
done |tee MovieTitles.txt
```

### Extracting IMDBid of Movie using Title, also obtain movie genre
```
lasttitle="";
MDFILE=metadata/moviescenes-metadata.json;
for item in `seq 1 22450`; do  
  id=`jq .[$item].YouTubeID $MDFILE`;
  mt=`jq .[$item].IMDBTitle $MDFILE`; 
  if [[ $mt != $lasttitle ]]; then 
    imdbid=`imdbpy search movie -n 1 "$mt"|tail -1|cut -d' ' -f4`; 
    g=`imdbpy get movie $imdbid|grep -i "genre"|sed 's/Genres://g'`;
  fi; 
  echo $id,$imdbid,$mt,\"$g\"; 
  lasttitle=$mt; 
done > >(tee metadata/YouTubeID-IMDBID-genre.csv)
```
### Getting Number of frames, resolution 
```
for i in `ls videos/*.mp4`; do 
  ID=`basename $i|sed -e 's/YTID_//g' -e s/\.mp4//g`;
  nframes=`ffmpeg -i $i -map 0:v:0 -c copy -f null - 2>&1|grep "frame="|cut -d' ' -f2`;
  if [ -z $nframes ]; then
    nframes=`ffmpeg -i $i -vcodec copy -f rawvideo -y /dev/null 2>&1|tr ^M '\n'|awk '/^frame=/ {print $2}'|tail -n 1`
  fi 
  
  res=`ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $i`; 
  echo "$ID",$nframes,$res; 
done > >(tee moviescenes-metadata.csv)
```
### Getting statistics viewcount, likecount, dislikecount, comments
```
for id in `cut -c1-11  metadata/moviescenes-metadata-Titles.tsv`; do 
  x=`grep -e "$id" metadata/moviescenes-metadata-Statistics.tsv|sed 's/\t/,/g'`; 
  if [ -z "$x" ]; then 
    x="$id,,,"; 
  fi; 
  echo $x;
done > >(tee rel-Titles-statistics.csv
```
# YouTube Video Scraper and IMDb Metadata Fetcher

This repository contains two Python scripts for scraping YouTube video data (including comments and metadata) and fetching IMDb metadata for movies. Below is an overview of each script and how to use them.

---

## 1. **YouTube Video Scraper (`YouTubeVideoScraper.py`)**

This script allows you to:
- Download YouTube video metadata (title, tags, view count, likes, dislikes, comments, etc.).
- Download comments from a YouTube video.
- Download the video itself in the highest available resolution.

### Features:
- **YouTube API Integration**: Fetches video metadata using the YouTube Data API.
- **Comment Scraping**: Scrapes comments from YouTube videos using web scraping techniques.
- **Video Download**: Downloads the video using the `pytube` library.

### Usage:

#### Command-Line Arguments:
- `--youtubeid` or `-y`: The YouTube video ID (required).
- `--output` or `-o`: The output directory to save results (required).
- `--developer_key` or `-k`: Your YouTube Data API developer key (required).
- `--limit` or `-l`: Limit the number of comments to download (optional).
- `--download_video` or `-d`: Download the video (optional flag).

#### Example:
```bash
python YouTubeVideoScraper.py --youtubeid <VIDEO_ID> --output ./output --developer_key <API_KEY> --limit 100 --download_video
```

This will:
1. Download the video metadata and save it to `video_info.json`.
2. Download up to 100 comments and save them to `comments.json`.
3. Download the video and save it as `YTID_<VIDEO_ID>.mp4` in the output directory.

---

## 2. **IMDb Metadata Fetcher (`getMovieInfo.py`)**

This script allows you to:
- Fetch IMDb metadata (movie ID, title, year) for movies listed in a file or for a specific movie title and year.

### Features:
- **IMDb Integration**: Uses the `imdbpy` library to search for movie metadata.
- **File Support**: Can process a file containing a list of movie titles and years.
- **Direct Query**: Can fetch metadata for a specific movie title and year.

### Usage:

#### Command-Line Arguments:
- `--file` or `-f`: Path to a metadata index file containing movie titles and years (optional).
- `--title` or `-t`: Movie title (required if `--file` is not provided).
- `--year` or `-y`: Movie year (optional, used with `--title`).

#### Example 1: Fetch metadata for movies listed in a file
```bash
python getMovieInfo.py --file ./movies.txt
```

#### Example 2: Fetch metadata for a specific movie
```bash
python getMovieInfo.py --title "Inception" --year 2010
```

This will print the movie's year, title, and IMDb ID (or `IMDB_INFO_NA` if not found).

---

## File Descriptions

### `YouTubeVideoScraper.py`
- **Purpose**: Scrapes YouTube video metadata, comments, and downloads videos.
- **Input**: YouTube video ID, output directory, and API key.
- **Output**: Video metadata (`video_info.json`), comments (`comments.json`), and the video file.

### `getMovieInfo.py`
- **Purpose**: Fetches IMDb metadata for movies.
- **Input**: Either a file containing movie titles or a specific movie title and year.
- **Output**: Prints IMDb metadata (year, title, and movie ID) to the console.

---

## Example Workflow

1. **Download YouTube Video Metadata and Comments**:
   ```bash
   python YouTubeVideoScraper.py --youtubeid dQw4w9WgXcQ --output ./youtube_data --developer_key <API_KEY> --limit 50
   ```

2. **Fetch IMDb Metadata for Movies**:
   ```bash
   python getMovieInfo.py --file ./movies.txt
   ```

3. **Combine Data**:
   - Use the IMDb metadata to enrich the YouTube video data (e.g., match movie titles to YouTube video titles).

---
