#! /bin/bash

echo "초기 실행에 30분정도 소요됩니다."
echo "초기 실행에 30분정도 소요됩니다."
echo "초기 실행에 30분정도 소요됩니다."
echo "초기 실행에 30분정도 소요됩니다."
echo "초기 실행에 30분정도 소요됩니다."

cd ../manifests/

while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
