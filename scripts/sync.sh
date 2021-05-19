rsync -a -v --delete  RI021:~/myproject/demo2/face_recognition/src/mlflow face_recognition/src
rsync -a -v --delete --exclude 'face_recognition/input' ../demo2 RI021:~/myproject/
# 逆
# rsync -a -v --delete  --exclude 'face_recognition/input' RI021:~/myproject/demo2/  .
# rsync -a -v --delete RI021:~/myproject/demo2/face_recognition/input face_recognition/