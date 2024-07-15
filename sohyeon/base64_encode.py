import base64

# 이미지 파일 경로
image_path = "/mnt/c/Users/BIG03-01/eduFocus/sohyeon/frame.jpg"

# 이미지 파일을 읽어서 Base64로 인코딩
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

# 인코딩된 문자열 출력
print(encoded_string)
