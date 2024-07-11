#!/bin/bash

### install go-lang ###

wget https://golang.org/dl/go1.22.4.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.4.linux-amd64.tar.gz
rm go1.22.4.linux-amd64.tar.gz

echo 'export GOROOT=/usr/local/go' >> ~/.bashrc
echo 'export GOPATH=$HOME/go' >> ~/.bashrc
echo 'export PATH=$PATH:$GOROOT/bin:$GOPATH/bin' >> ~/.bashrc
source ~/.bashrc
go version

### install Kustomize ### 
set -e

curl --silent --location --remote-name "https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize/v3.2.3/kustomize_kustomize.v3.2.3_linux_amd64"
chmod a+x kustomize_kustomize.v3.2.3_linux_amd64
sudo mv kustomize_kustomize.v3.2.3_linux_amd64 /usr/local/bin/kustomize

kustomize version
