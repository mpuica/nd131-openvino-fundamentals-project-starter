#!/bin/bash

#start the app 
echo "Start app"
source /opt/intel/openvino/bin/setupvars.sh
python3 main.py -d MYRIAD -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_mobilnet_v2_coco_2018_03_09_ir7/ssd_mobilnet_v2_coco_2018_03_09_ir7.xml -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
