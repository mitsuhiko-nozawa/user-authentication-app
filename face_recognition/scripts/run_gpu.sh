sh scripts/sync.sh
#ssh RI021 "mkdir ~/myproject/demo2/face_recognition/src/experiments/exp_$1"
#scp -r training/experiments/exp_$1/*.sh RI021:~/myproject/11_demospace/training/experiments/exp_$1 # experiment dir



if [ $3 ]; then
    echo pidあり
    ssh RI021 <<EOC
    cd ~/myproject/demo2
    docker start demospace_mnozawa
    nohup sh -c 'while ps -p $3 > /dev/null; do sleep 60; done; docker exec -d demospace_mnozawa  sh /workspace/face_recognition/scripts/run_exp.sh $1 $2' &
    exit
EOC

else 
    echo pid なし
    ssh RI021 <<EOC
    cd ~/myproject/demo2
    docker start demospace_mnozawa
    docker exec demospace_mnozawa  sh /workspace/face_recognition/scripts/run_exp.sh $1 $2
    exit
EOC
fi
