#!/bin/bash

#start ffmpeg server
echo "Start FFmpeg server"
sudo ffserver -f ./ffmpeg/server.conf
