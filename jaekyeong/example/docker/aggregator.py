from flask import Flask, request
import requests

app = Flask(__name__)
classification_url = "http://classification-model-service/classify"

@app.route('/aggregate', methods=['POST'])
def aggregate():
    result = request.json
    # 결과 집계 로직
    aggregated_result = aggregate_results(result)
    response = requests.post(classification_url, json=aggregated_result)
    return response.json(), response.status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
