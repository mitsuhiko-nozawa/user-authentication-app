# $1 exp number
# $2 d or f
echo cpu training

docker start demospace_mnozawa
#docker exec demospace_mnozawa ls
docker exec demospace_mnozawa sh /workspace/face_recognition/scripts/run_exp.sh $1 $2