### 설명 ###

#### Minio

> minio -> k port-forward service/minio-service 9001:9001          
> http://127.0.0.1:9001 접속         
> id = minio        
> password = minio123          

> column = [ip_address, video_id, blink, expression, gaze, time_stamp]



#### source local_storage.sh 실행시 container에서 local로 data를 복사해옴

#### source run_agg.sh 실행시 image빌드부터 cluster에 container deploy까지 작업