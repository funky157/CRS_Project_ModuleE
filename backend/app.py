# =====================================================
# CRS PROJECT
# File: app.py
# Purpose:
#   Flask API layer for CRS recommender
#
# Handles:
#   - Concept explanation
#   - Related topics
#
# Author: Harish
# =====================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import explain

app = Flask(__name__)
CORS(app)


@app.route("/recommend", methods=["POST"])
def recommend_api():
    try:
        data = request.get_json(force=True)

        topic = data.get("topic", "").strip()
        time_minutes = int(data.get("time_minutes", 5))

        if not topic:
            return jsonify({
                "error": "Topic is required",
                "explanation": "",
                "related_topics": []
            }), 400

        result = explain(topic, time_minutes)

        return jsonify({
            "topic": topic,
            "time_minutes": time_minutes,
            "explanation": result.get("explanation", ""),
            "related_topics": result.get("related_topics", [])
        })

    except Exception as e:
        print("[API ERROR]:", e)
        return jsonify({
            "error": "Internal server error",
            "explanation": "",
            "related_topics": []
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True)
