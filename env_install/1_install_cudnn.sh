#!/bin/bash
### linux 기반 cuda/ cudnn 설치 스크립트입니다. ###
### 이미 설치 되어있다면 실행하지 마세요. ###
### 버전은 gpu에 맞는 버전으로 변경해주세요 (CUDNN_VERSION, CUDA_VERSION, CUDNN_TAR_FILE)
set -e

sudo apt-get -y update
sudo apt-get -y remove --purge 'cuda-.*'
sudo apt-get -y install nvidia-cuda-toolkit

nvcc -V
whereis cuda

mkdir -p ~/nvidia
cd ~/nvidia

CUDNN_VERSION="8.3.3.40"
CUDA_VERSION="11.5"
CUDNN_TAR_FILE="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda${CUDA_VERSION}-archive.tar.xz"

wget https://developer.download.nvidia.com/compute/redist/cudnn/v${CUDNN_VERSION}/local_installers/${CUDA_VERSION}/${CUDNN_TAR_FILE}
tar -xf ${CUDNN_TAR_FILE}
mv cudnn-linux-x86_64-${CUDNN_VERSION}_cuda${CUDA_VERSION}-archive cuda

### cuda toolkit이 /usr/local/ 디렉토리 하위에 설치되는것을 가정으로 작성되었습니다. ###
### 만약 /usr/lib/ 디렉토리 하위에 cuda toolkit이 설치되었다면 아래의 경로를 수정해주세요. ###
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include/
sudo cp cuda/lib/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
sudo chmod a+r /usr/local/cuda/include/cudnn*.h

echo "export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
export "LD_LIBRARY_PATH=/usr/lib/cuda/include:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

sudo apt-get install -y nvidia-driver-460

echo "Please reboot system"
