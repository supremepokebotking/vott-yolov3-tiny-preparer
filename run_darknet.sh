# /bin/bash

mkdir -p workspace
cp -R training_data/vott-csv-export workspace
ls workspace
ls workspace/vott-csv-export

mv darknet53.conv.74 workspace/vott-csv-export/darknet53.conv.74
python3 prepare_for_darknet_pub.pyc
ls training_data

cd workspace/vott-csv-export
mkdir backup
ls
darknet detector train obj.data yolov3-tiny.cfg darknet53.conv.74
ls backup

zip -r /training_data/vott-csv-export/result.zip obj.names obj.data yolov3-tiny.cfg backup/yolov3-tiny_final.weights
