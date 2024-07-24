# MongoDB Setting
## 1. 데이터 준비(save_saliency 디렉터리에서 실행)
1. save_saliency/contents/contents1 경로에 컨텐츠 영상 파일(mp4) 저장
2. 명령어를 실행하여 mp4파일을 20FPS의 이미지로 변환
    ```
    source video_to_image.sh contents1
    ```

## 2. Saliency map 생성(save_saliency 디렉터리에서 실행)
1. 명령어를 실행하여 Saliency map 생성에 필요한 라이브러리 설치
    ```
    pip install -r requirements.txt
    ```
2. 명령어를 실행하여 save_saliency/contents/contents1에 저장된 이미지들의 Saliency map을 생성하여 save_saliency/saliency_contents/contents1에 저장
    ```
    python3 main.py contents1
    ```

## 3. mongodb_pv.yaml 파일 수정
1. 명령어를 실행하여 나온 결과(IP)를 nfs의 server에 입력
    ```
    ip route | grep eth0 | grep src | sed -n 's/.*src \([0-9\.]*\).*/\1/p'
    ```
2. NFS와 연결할 로컬의 디렉터리 경로를 nfs의 path에 입력
    ```
    /home/<USER NAME>/eduFocus/bok/save_saliency/saliency_contents
    ```

## 4. MongoDB 및 Express pod 실행
1. 명령어를 실행하여 Kubernetes에 Pod 생성
    ```
    source apply.sh
    ```

## 5. MongoDB에 데이터 저장
1. 명령어를 실행하여 마운트된 데이터를 MongoDB에 저장
    ```
    source mongodb_setting.sh
    ```

## 5. MongoDB 컬렉션에 데이터 넣기
- 명령어를 실행하여 MongoDB Pod에 접속
    ```
    kubectl exec -it $MONGODB_POD -- bash
    ```
- Pod 내부에서 명령어를 실행하여 MongoDB로 데이터 이동
    ```
    for FILE in /data/db/contents1/frame_*.bson; do mongorestore --db=saliency_db --collection=contents1 ${FILE}; done
    ```

## Mongo-express로 MongoDB에 저장된 데이터 확인 방법
```
MONGO_EXPRESS=$(k get pods -l app=mongo-express -o=jsonpath='{.items[0].metadata.name}')
kubectl port-forward $MONGO_EXPRESS 8081:8081
```

# Saliency Map 원리
- **총 4개의 특징맵으로 Saliency map 생성**
    1. 강도 특징 맵: 이미지를 가우시안 피라미드로 피라미드 이미지의 중심과 주변의 차이를 계산
    2. 색상 특징 맵: 피라미드 이미지의 RGB값의 차이를 계산
    3. 방향 특징 맵: 피라미드에 0,45,90,135도 방향의 Gabor Filter를 적용하여 각 방향의 중심과 주변의 차이를 계산
    4. 움직임 특징 맵: OpenCV의 OpticalFlow를 활용하여 이전 프레임과의 움직임 차이 계산
- **4개의 특징맵에 각각 가중치를 부여하여 나온 결과값 합산**
- **정규화 진행**
- **노이즈 제거를 위해 양방향 필터 적용**
### pySaliencyMapDefs.py에서 파라미터 수정하여 컨텐츠별 Saliency map 생성 가능