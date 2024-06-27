## kubectl을 설치하는 방법입니다. 아직 미완이니 실행하지 마세요!!!!

# 1. Kubectl 최신버전 설치
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
# 2. 바이너리 유효성 검사(체크섬 파일) 다운로드
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"
# 3. kubectl 바이너리의 유효성 검사
echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check
# 4. kubectl 설치
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
# 5. 설치한 버전이 최신인지 테스트
kubectl version --client
# 6. 저장소를 사용하는 데 필요한 패키지 설치 및 업데이트
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg
# 7. 공개 서명 키 다운로드
sudo mkdir -p -m 755 /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.30/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
sudo chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg
# 8. apt 저장소 추가
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.30/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo chmod 644 /etc/apt/sources.list.d/kubernetes.list
# 9. 패키지 인덱스를 업데이트한 후 apt 저장소를 추가한 kubectl 설치
sudo apt-get update
sudo apt-get install -y kubectl
# 10. bash 완성 설치
source /usr/share/bash-completion/bash_completion
# 11. kubectl 자동 완성 활성화
echo 'source <(kubectl completion bash)' >>~/.bashrc
# 12. kubectl을 k로 별칭
echo 'alias k=kubectl' >>~/.bashrc
echo 'complete -o default -F __start_kubectl k' >>~/.bashrc
# 13. bashrc 실행
source ~/.bashrc
# 14. kubectl, kubectl.sha256파일 삭제
rm kubectl kubectl.sha256