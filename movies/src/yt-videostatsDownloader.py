import os
import requests
import argparse
import numpy as np
import json
import datetime
from calendar import timegm
from pytube import YouTube
import time

import logging

MOVIECLIPS_CHANNEL_ID = "UC3gNmTGu-TTbFPpfSs5kNkg"
MOVIECLIPS_TRAILERS_ID = "UCi8e0iOVk1fEOogdfu4YgfA"

DEVELOPER_KEY = 'AIzaSyDL9Su1eLY5r2x9aLiVZsz0oObksZ-3ry0'
YOUTUBE_API_URL = 'https://www.googleapis.com/youtube/v3/'
VIDEO_API = 'videos?part=snippet,statistics&key={}&id={}'
SEARCH_API = 'search?part=snippet,id&order=date&maxResults=100&key={}&channelId={}'

STAT_DB = "__stats.csv"
DESC_FILE = "TTD_{}.json"
VIDEO_ID_FILE = "movie-scene-ids.txt"
MOVIE_TRAILER_LIST = "movie-trailer-ids.txt"


def download_videos(id_list=None, output_dir=None):
    for youtube_id in id_list:
        _of = os.path.join(output_dir, ("YTID_%s.mp4" % youtube_id))
        _url = 'https://youtu.be/%s' % youtube_id
        _yt = YouTube(url=_url, allow_oauth_cache=True)
        _dl = _yt.streams.get_highest_resolution()
        _dl.download(output_path=_of)


def get_all_video_in_channel(output_dir):
    channel_id = MOVIECLIPS_TRAILERS_ID
    url = YOUTUBE_API_URL + SEARCH_API.format(DEVELOPER_KEY, channel_id)
    video_ids = []
    filename = os.path.join(output_dir, MOVIE_TRAILER_LIST)
    continue_download = True

    logging.info('Saving video ids in file %s' % filename)
    fp = open(filename, "w")

    while continue_download:
        # Invoke end-point service call
        response = requests.get(url)
        for result in response.json().get('items', []):
            if result['id']['kind'] == "youtube#video":
                video_id = result['id']['videoId']
                video_ids.append(video_id)
                fp.write("%s\n" % video_id)
                logging.info('Retrieved YouTube ID %s' % video_id)
            try:
                next_page_token = result['nextPageToken']
                url = url + '&pageToken={}'.format(next_page_token)
            except:
                continue_download = False
                break
            if video_ids.len() % 10 == 0:
                logging.info("Download progress: [%d]th video ID [%s]" % (video_ids.len(), video_id))
    fp.close()


def get_video_info(youtube_ids, output_dir):
    pref = datetime.datetime.fromtimestamp(int(timegm(time.gmtime()))).strftime('%Y-%m-%d_%H:%M:%S')
    statfile_path = os.path.join(output_dir, pref + STAT_DB)
    fp = open(statfile_path, "w")

    for youtube_id in youtube_ids:
        desc_file_path = os.path.join(output_dir, DESC_FILE.format(youtube_id))
        logging.info('Getting contents for YouTube video [%s]' % youtube_id)
        # Invoke end-point service call
        url = YOUTUBE_API_URL + VIDEO_API.format(DEVELOPER_KEY, youtube_id)
        response = requests.get(url)

        desc = {}
        for result in response.json().get('items', []):

            desc['Title'] = result['snippet']['title']
            desc['Tags'] = result['snippet']['tags'] if 'tags' in result['snippet'] else {}
            if desc['Tags']:
                desc['MovieName'] = desc['Tags'][0]
            desc['Description'] = result['snippet']['description'] if 'description' in result['snippet'] else {}

            view_count = int(result['statistics']['viewCount']) if 'viewCount' in result['statistics'] else 0
            desc['ViewCount'] = view_count

            like_count = int(result['statistics']['likeCount']) if 'likeCount' in result['statistics'] else 0
            desc['LikeCount'] = like_count

            dislike_count = int(result['statistics']['dislikeCount']) if 'dislikeCount' in result['statistics'] else 0
            desc['DislikeCount'] = dislike_count

            comment_count = int(result['statistics']['commentCount']) if 'commentCount' in result['statistics'] else 0
            desc['CommentCount'] = comment_count

            fav_count = int(result['statistics']['favoriteCount']) if 'favoriteCount' in result['statistics'] else 0

            logging.info("%s,%d,%d,%d,%d,%d\n" % (youtube_id, view_count, like_count, dislike_count, comment_count,
                                                  fav_count))
            fp.write("%s,%d,%d,%d,%d,%d\n" % (youtube_id, view_count, like_count, dislike_count, comment_count,
                                              fav_count))
            logging.info('Updated %s file' % statfile_path)
            with open(desc_file_path, 'w') as outfile:
                json.dump(desc, outfile)
            logging.info('Written contents for YouTube video [%s] in [%s]' % (youtube_id, desc_file_path))
    fp.close()


def get_movie_info(youtube_id):
    print('INF: Getting information for movie corresponding to YouTube video [%s]' % youtube_id)
    # Invoke end-point service call
    filter_list = ['- Official Trailer', 'Trailer', 'HD', 'Official International Trailer', 'Movie Trailer HD']

    url = YOUTUBE_API_URL + VIDEO_API.format(DEVELOPER_KEY, youtube_id)
    response = requests.get(url)

    for result in response.json().get('items', []):
        ttl = result['snippet']['title']
        

def run():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--list_id', dest="list_id", type=str, required=True, help="List of YouTube video IDs")
    p.add_argument('-o', '--output_dir', dest="output_dir", type=str, required=True, help="Directory to store results")

    args = p.parse_args()

    if not args.list_id:
        p.print_help()
    else:
        if not os.path.exists(args.list_id):
            logging.info("Input video list file %s does not exist" % args.list_id)
            return 1
    ids = list(np.genfromtxt(args.list_id, dtype='str'))
    if len(ids) == 0:
        logging.info("Empty ID list in %s" % args.list_id)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    get_video_info(ids, args.output_dir)


def run2():
    p = argparse.ArgumentParser()
    p.add_argument('-o', '--output_dir', dest="output_dir", type=str, required=True, help="Directory to store results")
    args = p.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    get_all_video_in_channel(args.output_dir)


if __name__ == "__main__":
    run()
