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