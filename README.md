## 프로젝트 소개


이미지 처리를 이용한 유아의 집중도 확인
- 얼굴 표정을 활용
- 얼굴의 눈 깜빡임 활용
- 얼굴의 눈의 움직임을 활용



## 팀원 소개
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/JooBok">
        <img src="https://github.com/JooBok.png" width="150px;" alt="주복"/>
        <br />
        <sub><b>👑 이주복</b><br>🙋‍♂️ 팀장</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/pch229">
        <img src="https://github.com/pch229.png" width="150px;" alt="찬혁"/>
        <br />
        <sub><b>박찬혁</b><br>🙋‍♂️ 데이터 파이프라인 구축</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/sooooohyeon">
        <img src="https://github.com/sooooohyeon.png" width="150px;" alt="소현"/>
        <br />
        <sub><b>전소현</b><br>🙋‍♀️ 모델 구축</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Pepi10">
        <img src="https://github.com/Pepi10.png" width="150px;" alt="재경"/>
        <br />
        <sub><b>김재경</b><br>🙋‍♂️ 모델 구축, 배포</sub>
      </a>
    </td>
</table>


## 기술 스택

<img src="https://img.shields.io/badge/Apache Kafka-%3333333.svg?style=for-the-badge&logo=Apache Kafka&logoColor=white"> <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white"> 
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=Flask&logoColor=white"> <img src="https://img.shields.io/badge/kubernetes-123456?style=for-the-badge&logo=kubernetes&logoColor=white"> <img src="https://img.shields.io/badge/OpenCv-k73aba?style=for-the-badge&logo=OpenCv&logoColor=white"> <img src="https://img.shields.io/badge/TensorFlow-4538ff?style=for-the-badge&logo=TensorFlow&logoColor=white"> <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=Linux&logoColor=white"> <img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=Ubuntu&logoColor=white"> <img src="https://img.shields.io/badge/Rest api-ff38db?style=for-the-badge&logo=Rest api&logoColor=white"> <img src="https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=MongoDB&logoColor=white"> <img src="https://img.shields.io/badge/Git-06D6A9?style=for-the-badge&logo=Git&logoColor=white"> <img src="https://img.shields.io/badge/Github-181717?style=for-the-badge&logo=Github&logoColor=white"> <img src="https://img.shields.io/badge/Dlib-008000?style=for-the-badge&logo=Dlib&logoColor=white"> <img src="https://img.shields.io/badge/cmake-064F8C?style=for-the-badge&logo=cmake&logoColor=white"> <img src="https://img.shields.io/badge/numpy-0093DD?style=for-the-badge&logo=numpy&logoColor=white"> <img src="https://img.shields.io/badge/keras-b30000?style=for-the-badge&logo=keras&logoColor=white"> <img src="https://img.shields.io/badge/local-7FEE64?style=for-the-badge&logo=local&logoColor=white"> <img src="https://img.shields.io/badge/virtualbox-8BC0D0?style=for-the-badge&logo=virtualbox&logoColor=white"> <img src="https://img.shields.io/badge/anaconda-FFA116?style=for-the-badge&logo=anaconda&logoColor=white"> <img src="https://img.shields.io/badge/redis-FF4438?style=for-the-badge&logo=redis&logoColor=white">
------
⏩ 실행방법
- env_install 디렉터리의 셸파일들을 순서대로 하나씩 실행합니다.
- run_v2 디렉터리에서 run_v2.sh 파일을 실행합니다.
- bok 디렉터리에서 README 내용을 참고하여 순차적으로 진행합니다.
- 모든 쿠버네티스의 파드가 생성되고, MongoDB에 Saliency map이 저장이 완료되었는지 확인합니다.
- 모든 설정이 완료되었습니다.
-----
❗ 주의사항
- 현재 태블릿에서 얼굴 영상 데이터를 로컬 PC로 불러온 후 마지막 프레임 트리거가 설정되어 있지 않아 실제 구동은 불가능합니다.
- 테스트 얼굴 데이터 영상과 테스트 컨텐츠 영상을 사용하여 결과를 테스트해볼 수 있습니다.
- 테스트 시 run_v2 디렉터리의 test_api.py파일을 실행하여 테스트할 수 있습니다.
