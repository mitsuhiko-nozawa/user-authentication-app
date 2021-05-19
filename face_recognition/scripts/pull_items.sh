cd `dirname $0`
cd ../
rsync -a -v --delete RI021:~/myproject/demo2/face_recognition/src/experiments/exp_$1  src/experiments/
