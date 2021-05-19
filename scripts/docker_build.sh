docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t demospace_mnozawa -f docker/cpu.Dockerfile .
docker create -it --shm-size=8g --name demospace_mnozawa -v $(pwd):/workspace demospace_mnozawa
