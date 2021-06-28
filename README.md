# User Authentication App 
センサーデータを用いない、カメラによる入力画像のみを使った顔認証システムのデモアプリを作りました。

mtcnn
https://github.com/ipazc/mtcnn

### useful commands
```
# sync remote server
sh scripts/sync.sh

# build docker image
sh scripts/docker_build.sh

# serve gui application(jupyter, tensorboard, streamlit-app)
sh serve_jupyter.sh
sh serve_tensorboard face_recognition exp_xxx
sh serve_app 
```
### run experiments
```
# d means debug, f means full

# cpu
sh face-recognition/scripts/run_cpu.sh {exp number} {d or f}

# gpu
sh face-recognition/scripts/run_cpu.sh {exp number} {d or f}

# pull experiment result from remote server
sh face-recognition/scripts/pull_items.sh {exp number}
```