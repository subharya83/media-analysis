#!/usr/bin/env python

import os
import sys
import time
import json
import requests
import argparse
import lxml.html
import logging
import numpy as np
import datetime
from calendar import timegm
from pytube import YouTube
from lxml.cssselect import CSSSelector

class YouTubeVideoScraper:
    YOUTUBE_COMMENTS_URL = 'https://www.youtube.com/all_comments?v={youtube_id}'
    YOUTUBE_COMMENTS_AJAX_URL = 'https://www.youtube.com/comment_ajax'
    YOUTUBE_API_URL = 'https://www.googleapis.com/youtube/v3/'
    VIDEO_API = 'videos?part=snippet,statistics&key={}&id={}'
    SEARCH_API = 'search?part=snippet,id&order=date&maxResults=100&key={}&channelId={}'
    USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36'

    def __init__(self, developer_key=None):
        self.developer_key = developer_key
        self.session = requests.Session()
        self.session.headers['User-Agent'] = self.USER_AGENT

    def find_value(self, html, key, num_chars=2):
        pos_begin = html.find(key) + len(key) + num_chars
        pos_end = html.find('"', pos_begin)
        return html[pos_begin: pos_end]

    def extract_comments(self, html):
        tree = lxml.html.fromstring(html)
        item_sel = CSSSelector('.comment-item')
        text_sel = CSSSelector('.comment-text-content')
        time_sel = CSSSelector('.time')
        author_sel = CSSSelector('.user-name')

        for item in item_sel(tree):
            yield {'cid': item.get('data-cid'),
                   'text': text_sel(item)[0].text_content(),
                   'time': time_sel(item)[0].text_content().strip(),
                   'author': author_sel(item)[0].text_content()}

    def extract_reply_cids(self, html):
        tree = lxml.html.fromstring(html)
        sel = CSSSelector('.comment-replies-header > .load-comments')
        return [i.get('data-cid') for i in sel(tree)]

    def ajax_request(self, url, params, data, retries=10, sleep=20):
        for _ in range(retries):
            response = self.session.post(url, params=params, data=data)
            if response.status_code == 200:
                response_dict = json.loads(response.text)
                return response_dict.get('page_token', None), response_dict['html_content']
            else:
                time.sleep(sleep)

    def download_comments(self, youtube_id, sleep=1):
        response = self.session.get(self.YOUTUBE_COMMENTS_URL.format(youtube_id=youtube_id))
        html = response.text
        reply_cids = self.extract_reply_cids(html)

        ret_cids = []
        for comment in self.extract_comments(html):
            ret_cids.append(comment['cid'])
            yield comment

        page_token = self.find_value(html, 'data-token')
        session_token = self.find_value(html, 'XSRF_TOKEN', 4)

        first_iteration = True

        while page_token:
            data = {'video_id': youtube_id,
                    'session_token': session_token}

            params = {'action_load_comments': 1,
                      'order_by_time': True,
                      'filter': youtube_id}

            if first_iteration:
                params['order_menu'] = True
            else:
                data['page_token'] = page_token

            response = self.ajax_request(self.YOUTUBE_COMMENTS_AJAX_URL, params, data)
            if not response:
                break

            page_token, html = response

            reply_cids += self.extract_reply_cids(html)
            for comment in self.extract_comments(html):
                if comment['cid'] not in ret_cids:
                    ret_cids.append(comment['cid'])
                    yield comment

            first_iteration = False
            time.sleep(sleep)

        for cid in reply_cids:
            data = {'comment_id': cid,
                    'video_id': youtube_id,
                    'can_reply': 1,
                    'session_token': session_token}

            params = {'action_load_replies': 1,
                      'order_by_time': True,
                      'filter': youtube_id,
                      'tab': 'inbox'}

            response = self.ajax_request(self.YOUTUBE_COMMENTS_AJAX_URL, params, data)
            if not response:
                break

            _, html = response

            for comment in self.extract_comments(html):
                if comment['cid'] not in ret_cids:
                    ret_cids.append(comment['cid'])
                    yield comment
            time.sleep(sleep)

    def get_video_info(self, youtube_id):
        url = self.YOUTUBE_API_URL + self.VIDEO_API.format(self.developer_key, youtube_id)
        response = self.session.get(url)
        desc = {}
        for result in response.json().get('items', []):
            desc['Title'] = result['snippet']['title']
            desc['Tags'] = result['snippet']['tags'] if 'tags' in result['snippet'] else {}
            desc['Description'] = result['snippet']['description'] if 'description' in result['snippet'] else {}
            desc['ViewCount'] = int(result['statistics']['viewCount']) if 'viewCount' in result['statistics'] else 0
            desc['LikeCount'] = int(result['statistics']['likeCount']) if 'likeCount' in result['statistics'] else 0
            desc['DislikeCount'] = int(result['statistics']['dislikeCount']) if 'dislikeCount' in result['statistics'] else 0
            desc['CommentCount'] = int(result['statistics']['commentCount']) if 'commentCount' in result['statistics'] else 0
            desc['FavoriteCount'] = int(result['statistics']['favoriteCount']) if 'favoriteCount' in result['statistics'] else 0
        return desc

    def download_video(self, youtube_id, output_dir):
        _of = os.path.join(output_dir, f"YTID_{youtube_id}.mp4")
        _url = f'https://youtu.be/{youtube_id}'
        _yt = YouTube(url=_url, allow_oauth_cache=True)
        _dl = _yt.streams.get_highest_resolution()
        _dl.download(output_path=_of)

def main():
    parser = argparse.ArgumentParser(description='Download YouTube comments and video statistics')
    parser.add_argument('--youtubeid', '-y', required=True, help='ID of YouTube video')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--developer_key', '-k', required=True, help='YouTube API developer key')
    parser.add_argument('--limit', '-l', type=int, help='Limit the number of comments')
    parser.add_argument('--download_video', '-d', action='store_true', help='Download the video')

    args = parser.parse_args()

    downloader = YouTubeVideoScraper(args.developer_key)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.download_video:
        downloader.download_video(args.youtubeid, args.output)

    video_info = downloader.get_video_info(args.youtubeid)
    with open(os.path.join(args.output, 'video_info.json'), 'w') as f:
        json.dump(video_info, f)

    count = 0
    with open(os.path.join(args.output, 'comments.json'), 'w') as f:
        for comment in downloader.download_comments(args.youtubeid):
            f.write(json.dumps(comment) + '\n')
            count += 1
            sys.stdout.write(f'INF: Downloaded {count} comment(s)\r')
            sys.stdout.flush()
            if args.limit and count >= args.limit:
                break

    print(f'INF: Downloaded {count} comments for {args.youtubeid}')

if __name__ == "__main__":
    main()
