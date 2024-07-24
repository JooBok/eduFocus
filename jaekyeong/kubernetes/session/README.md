## 개요
---

Redis를 사용하여 중앙 세션 관리를 구현하고 kubernetes로 배포하는 docker image를 만드는 디렉토리입니다.     
세션 생성, 조회, 업데이트 기능이 있으며, 각 세션은 5분 후 만료됩니다.     
API 게이트웨이 및 3개의 모델(gaze-tracking, blink-detection, emotion-analysis)과 상호작용하여 세션 데이터를 관리합니다.    

## 주요 구성 요소

1. [라이브러리 및 모듈](#1-라이브러리-및-모듈)
2. [Flask](#2-Flask)
3. [Redis](#3-Redis)
4. [세션 관리](#4-세션-관리)
5. [API 엔드포인트](#5-API-엔드포인트)

## 상세 설명

### 1. 라이브러리 및 모듈

* Flask: 웹 애플리케이션 프레임워크
* redis: Redis 데이터베이스 연결

### 2. Flask
Flask를 사용하여 API 서버를 구현

### 3. Redis
Redis를 사용하여 세션 데이터를 저장하고 관리

### 4. 세션 관리

* 세션 만료 시간: 5분    
* 세션 데이터 구조: Hash 형태의 IP 주소, 비디오 ID, 생성 시간, 마지막 접근 시간, 컴포넌트 데이터(gaze-tracking, blink-detection, emotion-analysis)

### 5. API 엔드포인트

* /mk-session (POST): 새로운 세션 생성    
* /get_session/<session_id> (GET): 세션 데이터 조회    
* /update_session/<session_id> (PUT): 세션 데이터 업데이트    

## 주요 함수

* mk_session()

    * 새 세션을 생성하고 Redis에 저장    
    * 이미 존재하는 세션인 경우 메시지를 반환    

* get_session(session_id: str)

    * 주어진 세션 ID에 해당하는 세션 데이터를 조회
    * 세션이 없는 경우 에러를 반환

* update_session(session_id: str)

    * 컴포넌트 세션 데이터를 업데이트
    * 세션의 마지막 접근 시간을 갱신
    * 세션이 없는 경우 에러를 반환

## 처리 흐름

* 세션 생성 요청 시:

    * 세션 ID 중복 확인
    * 새 세션 데이터 생성 및 Redis에 저장


* 세션 조회 요청 시:

    * Redis에서 세션 데이터 검색 및 반환


* 세션 업데이트 요청 시:

    * 기존 세션 데이터 검색
    * 요청된 컴포넌트의 데이터 업데이트
    * 마지막 접근 시간 갱신
    * 업데이트된 데이터를 Redis에 저장