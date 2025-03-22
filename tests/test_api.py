import requests

def test_api_predict():
    url = "http://127.0.0.1:8000/predict"
    data = {"text": "I had a terrible experience with Air Paradis."}
    response = requests.post(url, json=data)
    assert response.status_code == 200
    assert "sentiment" in response.json()