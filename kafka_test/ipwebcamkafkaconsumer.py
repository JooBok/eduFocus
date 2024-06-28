###################################################   original   ##########################################
# from kafka import KafkaConsumer
# import cv2
# import numpy as np

# # Kafka Consumer 설정
# consumer = KafkaConsumer(
#     'video-stream',  # 토픽 이름
#     bootstrap_servers=['localhost:9092'],  # Kafka 브로커 주소
#     auto_offset_reset='earliest',  # 가장 오래된 메시지부터 읽기 시작
#     enable_auto_commit=True,
#     group_id='video-consumer-group',
#     fetch_max_bytes=10485760, #10MB
#     max_partition_fetch_bytes=1048576 #1MB
# )

# print("Listening for messages on 'video-stream' topic...")

# frame_skip = 5
# frame_count = 0

# try:
#     print("1")
#     for message in consumer:
#         print("1-1-1")
#         frame_count += 1
#         print("1-1")
#         if frame_count % frame_skip != 0:
#                 print("2")
#                 continue

#         frame_data = message.value
#         nparr = np.frombuffer(frame_data, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
#         if frame is not None:
#             print("3")
#             cv2.imshow('Video Stream', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
        
#         cv2.destroyAllWindows()
#         print(f"Received message of size {len(message.value)} bytes")
#     if frame_count ==0:
#           print("No message available of size {len(message.value)} bytes")
# except KeyboardInterrupt:
#       print("\nStreaming stopped by user.")
# finally:
#       cv2.destroyAllWindows()
############################################################################################################################
from kafka import KafkaConsumer
import cv2
import numpy as np
import threading
from collections import deque

def consume_frames(consumer, frame_queue, skip_frames):
    frame_count = 0
    while True:
        print('Polling for messages...')
        message_pack = consumer.poll(timeout_ms=1000)  # 1초 동안 대기
        if not message_pack:
            print("No message available. Waiting for message...")
            continue

        for tp, messages in message_pack.items():
            for message in messages:
                frame_count += 1
                print(f"Received frame {frame_count}")
                if frame_count % skip_frames == 0:
                    frame_queue.append(message.value)
                    print(f"Appended from {frame_count} to queue")

def display_frames(frame_queue):
    while True:
        if frame_queue:
            frame_data = frame_queue.popleft()
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow('Video Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print("Frame queue is empty, waiting for frames...")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    kafka_topic = 'video-stream'
    kafka_bootstrap_servers = ['localhost:9092']
    skip_frames = 2  # 2로 설정하면 매 2번째 프레임만 처리

    # Kafka Consumer 설정
    consumer = KafkaConsumer(
        kafka_topic,  # 토픽 이름
        bootstrap_servers=kafka_bootstrap_servers,  # Kafka 브로커 주소
        auto_offset_reset='earliest',  # 가장 오래된 메시지부터 읽기 시작
        enable_auto_commit=True,
        group_id='video-consumer-group',  # 컨슈머 그룹 ID 설정
        fetch_max_bytes=10485760,  # 10MB
        max_partition_fetch_bytes=1048576,  # 1MB
        enable_auto_commit =True,
        api_version=(0, 2, 0)
    )

    print("Listening for messages on 'video-stream' topic...")

    frame_queue = deque(maxlen=10)

    consumer_thread = threading.Thread(target=consume_frames, args=(consumer, frame_queue, skip_frames))
    display_thread = threading.Thread(target=display_frames, args=(frame_queue,))

    consumer_thread.start()
    display_thread.start()

    try:
        consumer_thread.join()
        display_thread.join()
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        consumer.close()
        cv2.destroyAllWindows()

############################################################################################################################
# from kafka import KafkaConsumer
# import cv2
# import numpy as np
# # import threading

# consumer = KafkaConsumer(
#     'video-stream',
#     bootstrap_servers='localhost:9092',
#     auto_offset_reset='latest',
#     enable_auto_commit=True,
#     group_id='video-consumer-group',
#     consumer_timeout_ms=1000
# )

# for message in consumer:
#     frame_data = message.value
#     np_arr = np.frombuffer(frame_data, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     if frame is not None:
#         cv2.imshow('Video Stream', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# cv2.destroyAllWindows()
####################################################################################################################################
# from kafka import KafkaConsumer
# import cv2
# import numpy as np
# import threading
# from collections import deque

# def consume_frames(consumer, frame_queue, skip_frames):
#     frame_count = 0
#     for message in consumer:
#         frame_count += 1
#         if frame_count % skip_frames == 0:
#             frame_queue.append(message.value)

# def display_frames(frame_queue):
#     while True:
#         if frame_queue:
#             frame_data = frame_queue.popleft()
#             np_arr = np.frombuffer(frame_data, np.uint8)
#             frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#             if frame is not None:
#                 cv2.imshow('Video Stream', frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     kafka_topic = 'video-stream'
#     kafka_bootstrap_servers = ['localhost:9092']
#     skip_frames = 2  # 2로 설정하면 매 2번째 프레임만 처리

#     consumer = KafkaConsumer(
#         kafka_topic,
#         bootstrap_servers=kafka_bootstrap_servers,
#         auto_offset_reset='earliest',
#         enable_auto_commit=True,
#         group_id='video-consumer-group',
#         consumer_timeout_ms=1000
#     )

#     frame_queue = deque(maxlen=10)  # 최대 10개의 프레임을 큐에 저장

#     consumer_thread = threading.Thread(target=consume_frames, args=(consumer, frame_queue, skip_frames))
#     display_thread = threading.Thread(target=display_frames, args=(frame_queue,))

#     consumer_thread.start()
#     display_thread.start()

#     consumer_thread.join()
#     display_thread.join()

#     consumer.close()
######################################################################################################
######################################################################################################
# from kafka import KafkaConsumer
# import cv2
# import numpy as np
# import threading
# from collections import deque

# def consume_frames(consumer, frame_queue, skip_frames):
#     frame_count = 0
#     while True:
#         #새로운 메시지를 받기 위해 폴링
#         for message in consumer.poll(timeout_ms=1000):
#             frame_count += 1
#             if frame_count % skip_frames == 0:
#                 frame_queue.append(message.value)
#         # 메세지가 없는 경우 없다는 문구 출력
#         if frame_count == 0:
#             print("No message available. Waiting for message...")
#         frame_count = 0

# def display_frames(frame_queue):
#     while True:
#         if frame_queue:
#             frame_data = frame_queue.popleft()
#             np_arr = np.frombuffer(frame_data, np.uint8)
#             frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#             if frame is not None:
#                 cv2.imshow('Video Stream', frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     kafka_topic = 'video-stream'
#     kafka_bootstrap_servers = ['localhost:9092']
#     skip_frames = 2  # 2로 설정하면 매 2번째 프레임만 처리


# # Kafka Consumer 설정
# consumer = KafkaConsumer(
#     'video-stream',  # 토픽 이름
#     bootstrap_servers=['localhost:9092'],  # Kafka 브로커 주소
#     auto_offset_reset='earliest',  # 가장 오래된 메시지부터 읽기 시작
#     enable_auto_commit=True,
#     group_id='video-consumer-group',
#     fetch_max_bytes=10485760, #10MB
#     max_partition_fetch_bytes=1048576 #1MB
# )

# print("Listening for messages on 'video-stream' topic...")

# frame_queue = deque(maxlen=10)

# consumer_thread = threading.Thread(target=consume_frames, args=(consumer, frame_queue, skip_frames))
# display_thread = threading.Thread(target=display_frames, args=(frame_queue,))

# consumer_thread.start()
# display_thread.start()

# try:
#     while True:
#         pass
# except KeyboardInterrupt:
#     print("KeyboardInterrupt: Stopping threads...")
#     consumer.close()  # 컨슈머 종료
#     display_thread.join()  # 디스플레이 스레드 종료 대기
#     print("Threads stopped. Exiting main program.")
#     cv2.destroyAllWindows()
    #####################################################################################################################
# consumer.close()

# frame_skip = 5
# frame_count = 0

# for message in consumer:
#     frame_count += 1
#     if frame_count % frame_skip != 0:
#             continue

#     frame_data = message.value
#     nparr = np.frombuffer(frame_data, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
#     if frame is not None:
#         cv2.imshow('Video Stream', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     cv2.destroyAllWindows()
#     print(f"Received message of size {len(message.value)} bytes")