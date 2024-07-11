#!/bin/bash

### docker nvidia 최신버전 다운로드 스크립트입니다. ###
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get -y update
sudo apt-get -y install nvidia-docker2
sudo systemctl restart docker

sudo bash -c 'cat <<EOF > /etc/docker/daemon.json
{
    "exec-opts": ["native.cgroupdriver=systemd"],
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m"
    },
    "data-root": "/mnt/storage/docker_data",
    "storage-driver": "overlay2",
    "default-runtime" : "nvidia",
    "runtimes" : {
        "nvidia" : {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs" : []
        }
    }
}
EOF'
sudo systemctl restart docker
