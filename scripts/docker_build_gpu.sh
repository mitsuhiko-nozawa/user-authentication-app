docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t demospace_mnozawa -f docker/gpu.Dockerfile .
docker create -it --gpus '"device=0"' --shm-size=24g --name demospace_mnozawa -v $(pwd):/workspace demospace_mnozawa
