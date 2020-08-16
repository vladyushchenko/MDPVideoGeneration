#!/usr/bin/env bash

QUALITY=18
KEYFRAMES=1
WIDTH=320
HEIGHT=240

if [ ! "$#" -eq 2 ]; then
    echo "Not all arguments supplied..."
    echo "Need input directory, output directory  to process."
    read -rsp $'Press enter to exit...\n'
    exit 1
fi


INPUT_DIR="$1"
INPUT_DIR="${INPUT_DIR//\\//}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Directory does not exist or parameter is not directory."
    read -rsp $'Check / at the end...\n'
    exit 1
fi

OUTPUT_DIR="$2"
OUTPUT_DIR="${OUTPUT_DIR//\\//}"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Output directory is created!"
fi

echo "Input Directory: ${INPUT_DIR}"
VIDEO_EXT="\.webm$|\.flv$|\.vob$|\.ogg$|\.ogv$|\.drc$|\.gifv$|\.mng$|\.avi$|\.mov$|\.qt$|\.wmv$|\.yuv$|"
VIDEO_EXT="${VIDEO_EXT}\.rm$|\.rmvb$|/.asf$|\.amv$|\.mp4$|\.m4v$|\.mp*$|\.mkv$|\.svi$|\.3gp$|\.flv$|\.f4v$"

echo "Supported Extensions: ${VIDEO_EXT}"

INPUT_FILES=$(find "$INPUT_DIR" -type f | grep -E "$VIDEO_EXT" | sort -V )
echo 'Input files: '"$INPUT_FILES"

prevIFS="$IFS"
IFS=$'\n' read -rd '' -a ARR <<< "$INPUT_FILES"
ARR_LENGTH="${#ARR[@]}"

if [ "$ARR_LENGTH" -eq 0 ]; then
    echo "No files to process..."
    read -rsp $'Press enter to exit...\n'
    exit 1
fi

EXT='mp4'


for (( COUNTER=0; COUNTER<"$ARR_LENGTH"; COUNTER++ ));
do
    OUTPUT_NAME=${ARR[$COUNTER]##*/}
    OUTPUT_NAME=$( echo ${OUTPUT_NAME%.*} )
    OUTPUT_NAME="${OUTPUT_DIR}/${OUTPUT_NAME}_h264"
    OUTPUT_NAME=$(echo "$OUTPUT_NAME" | tr -d '\r')
    OUTPUT_NAME="${OUTPUT_NAME}.${EXT}"
    ffmpeg -i "${ARR[$COUNTER]}" -y -map v:0 -c:v libx264 -s "$WIDTH"x"$HEIGHT" -crf "$QUALITY" -pix_fmt yuv420p -g "$KEYFRAMES" -profile:v high "$OUTPUT_NAME"
    # ffmpeg -i "${ARR[$COUNTER]}" -y -map v:0 -c:v libx265 -s "$WIDTH"x"$HEIGHT" -preset ultrafast -x265-params keyint="$KEYFRAMES":no-open-gop=1:crf="$QUALITY":lossless=1  "$OUTPUT_NAME"
    wait
done

IFS="$prevIFS"
read -rsp $'Done, press key to exit...\n'
