
```
useful commands

# sync remote server
sh scripts/sync.sh

# build docker image
sh scripts/docker_build.sh
```

```
# run experiments
# d means debug, f means full

# cpu
sh face-recognition/scripts/run_cpu.sh {exp number} {d or f}

# gpu
sh face-recognition/scripts/run_cpu.sh {exp number} {d or f}
```


dockerコンテナ
学習用のコンテナは使い捨てない、
dockerコンテナはマシンとして扱うため、使い捨てるとプロセスIDなどがコンテナ間で見れない、ステートが存在する

他のコンテナ(ログを見る系、jupyter)などは使い捨ててもいいと思う