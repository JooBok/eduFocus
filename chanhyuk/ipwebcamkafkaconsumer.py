from kafka import KafkaConsumer
import cv2
import numpy as np
import threading
from collections import deque
import os

class FrameConsumer(threading.Thread):
    def __init__(self, consumer, frame_queue, stop_event):
        threading.Thread.__init__(self)
        self.consumer = consumer
        self.frame_queue = frame_queue
        # self.skip_frames = skip_frames
        self.stop_event = stop_event

    def run(self):
        frame_count = 0
        while not self.stop_event.is_set():
            print('Polling for messages...')
            message_pack = self.consumer.poll(timeout_ms=1000)
            if not message_pack:
                print("No message available. Waiting for message...")
                continue

            for tp, messages in message_pack.items():
                for message in messages:
                    frame_count += 1
                    print(f"Received frame {frame_count}")
                    # if frame_count % self.skip_frames == 0:
                    self.frame_queue.append((frame_count, message.value))
                    print(f"Appended frame {frame_count} to queue")

class FrameDisplay(threading.Thread):
    def __init__(self, frame_queue, save_path, stop_event):
        threading.Thread.__init__(self)
        self.frame_queue = frame_queue
        self.save_path = save_path
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            if self.frame_queue:
                frame_count, frame_data = self.frame_queue.popleft()
                np_arr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imshow('Video Stream', frame)

                    frame_filename = os.path.join(self.save_path, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    print(f"Saved frame {frame_count} to {frame_filename}")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop_event.set()
                        break
            else:
                print("Frame queue is empty, waiting for frames...")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    kafka_topic = 'video-stream'
    kafka_bootstrap_servers = ['localhost:9092']
    # skip_frames = 2
    save_path = './saved_frames'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    consumer = KafkaConsumer(
        kafka_topic,
        bootstrap_servers=kafka_bootstrap_servers,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='video-consumer-group',
        fetch_max_bytes=10485760,
        max_partition_fetch_bytes=1048576,
        api_version=(0, 2, 0)
    )

    print("Listening for messages on 'video-stream' topic...")

    frame_queue = deque(maxlen=10)
    stop_event = threading.Event()

    consumer_thread = FrameConsumer(consumer, frame_queue, stop_event)
    display_thread = FrameDisplay(frame_queue, save_path, stop_event)

    consumer_thread.daemon = True
    display_thread.daemon = True

    consumer_thread.start()
    display_thread.start()

    try:
        while not stop_event.is_set():
            pass
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
        stop_event.set()
    finally:
        consumer_thread.join()
        display_thread.join()
        consumer.close()
        cv2.destroyAllWindows()
