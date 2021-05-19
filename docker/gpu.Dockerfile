FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

# python install
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# nano wget curl
RUN apt-get -y install vim nano wget curl

# opencv lib packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# pip install
RUN pip install -U pip
RUN pip install --upgrade pip

# install pytorch gpu
RUN pip install torch==1.6.0 torchvision==0.7.0

# lycon 
RUN apt-get -y install cmake build-essential libjpeg-dev libpng-dev
RUN pip install lycon


COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# install git
RUN apt-get update && apt-get -y install git

# pycocotools
RUN git clone https://github.com/philferriere/cocoapi.git \
    && cd cocoapi/PythonAPI \
    && python setup.py build_ext install


ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID duser && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID duser && \
    adduser duser sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER duser



#WORKDIR /home/duser/
WORKDIR /workspace/