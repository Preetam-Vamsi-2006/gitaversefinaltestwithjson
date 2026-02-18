from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

FASTAPI_BASE = "http://127.0.0.1:8000"

@app.route("/get-meaning", methods=["POST"])
def get_meaning():
    data = request.json
    response = requests.post(
        f"{FASTAPI_BASE}/get-meaning",
        json=data
    )
    return jsonify(response.json())

@app.route("/audio/<filename>")
def serve_audio(filename):
    response = requests.get(
        f"{FASTAPI_BASE}/audio/{filename}",
        stream=True
    )
    return response.content, 200, {
        "Content-Type": "audio/mpeg"
    }

if __name__ == "__main__":
    app.run(port=5000, debug=True)
