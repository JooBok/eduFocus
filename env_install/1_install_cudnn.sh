#!/bin/bash
sudo apt-get -y update
sudo apt-get -y remove --purge 'cuda-.*'
sudo apt-get -y install nvidia-cuda-toolkit
nvcc -V
whereis cuda
mkdir ~/nvidia
cd ~/nvidia

CUDNN_TAR_FILE="cudnn-linux-x86_64-8.3.3.40_cuda11.5-archive.tar.xz"
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.3.3/local_installers/11.5/${CUDNN_TAR_FILE}
tar -xf ${CUDNN_TAR_FILE}
mv cudnn-linux-x86_64-8.3.3.40_cuda11.5-archive cuda

sudo cp cuda/include/cudnn.h /usr/lib/cuda/include/
sudo cp cuda/lib/libcudnn* /usr/lib/cuda/lib64/
sudo chmod a+r /usr/lib/cuda/lib64/libcudnn*
sudo chmod a+r /usr/lib/cuda/include/cudnn.h

echo "export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
export "LD_LIBRARY_PATH=/usr/lib/cuda/include:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

sudo apt-get install -y nvidia-driver-550
echo "reboot....."
