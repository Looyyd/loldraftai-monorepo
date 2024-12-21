import requests

input_data = {
    "champion_ids": [36, 5, 61, 147, 235, 163, 427, 910, 21, 111],
    "patch": "14.23",
    "numerical_elo": 1,
}

response = requests.post("http://localhost:8000/predict", json=input_data)
print(response.json())
