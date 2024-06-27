#!/bin/bash

### install go-lang ###

wget https://golang.org/dl/go1.22.4.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.4.linux-amd64.tar.gz
rm go1.22.4.linux-amd64.tar.gz

echo 'export GOROOT=/usr/local/go' >> ~/.bashrc
echo 'export GOPATH=$HOME/go' >> ~/.bashrc
echo 'export PATH=$PATH:$GOROOT/bin:$GOPATH/bin' >> ~/.bashrc

go version

### install Kustomize ### 
if [ ! -f /usr/local/bin/kusomize ]
  then
    echo "kustomize"
    wget https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize/v5.4.2/kustomize_v5.4.2_linux_amd64.tar.gz
    tar -xzf kustomize_v5.4.2_linux_amd64.tar.gz
    sudo chmod 777 kustomize
    sudo mv kustomize /usr/local/bin/kustomize
fi

kustomize version

source ~/.bashrc
