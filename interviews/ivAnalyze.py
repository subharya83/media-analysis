import argparse
import os
import moviepy.editor as mp
import speech_recognition as sr
from pyannote.audio import Pipeline
from deepface import DeepFace
import cv2
from datetime import timedelta

# Ensure weights directory exists
os.makedirs("weights", exist_ok=True)

def download_youtube_audio(video_url, output_audio_path):
    from pytube import YouTube
    yt = YouTube(video_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(filename=output_audio_path)

def extract_audio_from_video(video_path, output_audio_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio, show_all=True)
        return transcription['alternative'][0]['transcript'], transcription['alternative'][0]['confidence']
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None, None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None, None

def diarize_speakers(audio_path):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=True)
    diarization = pipeline(audio_path)
    return diarization

def analyze_facial_expressions(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    expressions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        timestamp = frame_count / fps
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if result:
            expressions.append((timestamp, result[0]['dominant_emotion']))
    cap.release()
    return expressions

def generate_ssml(transcription, diarization, expressions, output_ssml_path):
    with open(output_ssml_path, 'w') as ssml_file:
        ssml_file.write('<?xml version="1.0"?>\n')
        ssml_file.write('<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">\n')
        ssml_file.write('<p>\n')
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_time = segment.start
            end_time = segment.end
            ssml_file.write(f'<mark name="{speaker}" time="{start_time:.2f}"/>\n')
            ssml_file.write(f'<s>{transcription}</s>\n')
            for timestamp, emotion in expressions:
                if start_time <= timestamp <= end_time:
                    ssml_file.write(f'<mark name="{emotion}" time="{timestamp:.2f}"/>\n')
        ssml_file.write('</p>\n')
        ssml_file.write('</speak>\n')

def visualize_output(video_path, diarization, expressions, output_video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        timestamp = frame_count / fps

        active_speaker = None
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.start <= timestamp <= segment.end:
                active_speaker = speaker
                cv2.putText(frame, f"Speaker: {speaker}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if active_speaker:
            for ts, emotion in expressions:
                if abs(ts - timestamp) < 0.1:
                    cv2.putText(frame, f"Emotion: {emotion}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description="Analyze a YouTube video and generate an SSML file with time-coded speech segments, speaker labels, and facial expressions.")
    parser.add_argument('-i', '--input', required=True, help="Input YouTube video URL or local video file path.")
    parser.add_argument('-o', '--output', required=True, help="Output SSML file path.")
    parser.add_argument('-v', '--visualize', help="Output video file path for visualization.")
    args = parser.parse_args()

    input_path = args.input
    output_ssml_path = args.output
    output_video_path = args.visualize

    # Temporary audio file
    temp_audio_path = "temp_audio.wav"

    # Check if the input is a YouTube URL or a local file
    if input_path.startswith(('http://', 'https://')):
        print("Downloading audio from YouTube...")
        download_youtube_audio(input_path, temp_audio_path)
    else:
        print("Extracting audio from video file...")
        extract_audio_from_video(input_path, temp_audio_path)

    print("Transcribing audio...")
    transcription, confidence = transcribe_audio(temp_audio_path)
    if transcription:
        print(f"Transcription confidence: {confidence}")
        print("Diarizing speakers...")
        diarization = diarize_speakers(temp_audio_path)
        print("Analyzing facial expressions...")
        expressions = analyze_facial_expressions(input_path)
        print("Generating SSML file...")
        generate_ssml(transcription, diarization, expressions, output_ssml_path)
        print(f"SSML file generated at: {output_ssml_path}")

        if output_video_path:
            print("Generating visualization video...")
            visualize_output(input_path, diarization, expressions, output_video_path)
            print(f"Visualization video generated at: {output_video_path}")
    else:
        print("Transcription failed.")

    # Clean up temporary audio file
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

if __name__ == "__main__":
    main()