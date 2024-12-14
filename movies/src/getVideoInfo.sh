#### Getting Number of frames, resolution 
if [ $# -ne 2 ]; then
  echo "Usage <path/to/videos/> <path/to/output.csv>"
  exit
fi 

if [ ! -d $1 ]; then
  echo "$1 is not a valid directory"
  exit
fi 

VIDEO_DIR=$1;
OUTPUT_FILE=$2;

for i in `ls $VIDEO_DIR/*.mp4`; do 
  ID=`basename $i|sed -e 's/YTID_//g' -e s/\.mp4//g`;
  nframes=`ffmpeg -i $i -map 0:v:0 -c copy -f null - 2>&1|grep "frame="|cut -d' ' -f2`;
  if [ -z $nframes ]; then
    nframes=`ffmpeg -i $i -vcodec copy -f rawvideo -y /dev/null 2>&1|tr ^M '\n'|awk '/^frame=/ {print $2}'|tail -n 1`
  fi 
  
  res=`ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $i`; 
  echo "$ID",$nframes,$res; 
done > >(tee $OUTPUT_FILE)

