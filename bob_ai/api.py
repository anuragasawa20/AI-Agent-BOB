import os
from flask import Flask, request, jsonify
from main import WhiskyRecommendationSystem, BaxusAPI, load_collection_from_file
from dotenv import load_dotenv
import logging
import requests
import json
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize the recommendation system
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    )
    raise ValueError("OpenAI API key not found")

try:
    recommendation_system = WhiskyRecommendationSystem(
        data_path="../501 Bottle Dataset.csv", openai_api_key=api_key
    )
    logger.info("Whisky Recommendation System initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Whisky Recommendation System: {e}")
    raise


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return (
        jsonify(
            {
                "status": "healthy",
                "message": "Bob AI Whisky Recommendation System is running",
            }
        ),
        200,
    )


@app.route("/api/recommendations", methods=["POST"])
def get_recommendations():
    """
    Get whisky recommendations based on user's collection.

    Expected JSON payload:
    {
        "collection": [
            {
                "id": 123,
                "name": "Whisky Name",
                "spirit_type": "Bourbon",
                "avg_msrp": 50.0,
                "abv": 45.0,
                ...
            },
            ...
        ],
        "wishlist": [
            {
                "id": 456,
                "name": "Wishlist Whisky",
                "spirit_type": "Rye",
                ...
            },
            ...
        ],
        "max_recommendations": 5
    }
    """
    try:
        data = request.json

        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400

        collection = data.get("collection", [])
        wishlist = data.get("wishlist", [])
        max_recommendations = data.get("max_recommendations", 5)

        if not collection or not isinstance(collection, list):
            return jsonify({"error": "Collection must be a non-empty array"}), 400

        # Create recommendation request
        request_obj = WhiskyRecommendationSystem.RecommendationRequest(
            collection=collection,
            wishlist=wishlist,
            max_recommendations=max_recommendations,
        )

        # Get recommendations
        logger.info(
            f"Generating recommendations for collection with {len(collection)} items and {len(wishlist)} wishlist items"
        )
        response = recommendation_system.get_recommendations(request_obj)

        # Format response
        result = {
            "user_profile": response.user_profile,
            "recommendations": [
                {
                    "id": rec.id,
                    "name": rec.name,
                    "spirit_type": rec.spirit_type,
                    "price": rec.price,
                    "reasoning": rec.reasoning,
                    "similarity_score": rec.similarity_score,
                    "interesting_fact": rec.interesting_fact,
                }
                for rec in response.recommendations
            ],
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({"error": f"Failed to generate recommendations: {str(e)}"}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_collection():
    """
    Analyze a user's whisky collection and return the user profile.

    Expected JSON payload:
    {
        "collection": [
            {
                "id": 123,
                "name": "Whisky Name",
                "spirit_type": "Bourbon",
                "avg_msrp": 50.0,
                "abv": 45.0,
                ...
            },
            ...
        ]
    }
    """
    try:
        data = request.json

        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400

        collection = data.get("collection", [])

        if not collection or not isinstance(collection, list):
            return jsonify({"error": "Collection must be a non-empty array"}), 400

        # Extract user preferences
        user_preferences = recommendation_system._extract_user_preferences(collection)
        user_profile_text = recommendation_system._generate_user_profile(collection)

        # Format response
        result = {"user_profile": user_preferences, "profile_text": user_profile_text}

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error analyzing collection: {e}")
        return jsonify({"error": f"Failed to analyze collection: {str(e)}"}), 500


@app.route("/get/recommendation/<username>", methods=["GET"])
def get_recommendation_by_username(username):
    """
    Get whisky recommendations based on a user's Baxus username.
    This endpoint fetches the user's collection from the Baxus API and generates recommendations.

    Args:
        username: Baxus username to fetch collection for

    Returns:
        JSON response with recommendations and user profile
    """
    try:
        # Create recommendation request with username
        request_obj = WhiskyRecommendationSystem.RecommendationRequest(
            collection=[],  # Collection will be fetched from API using username
            username=username,
            max_recommendations=5,
        )

        # Get recommendations
        logger.info(f"Generating recommendations for user: {username}")
        response = recommendation_system.get_recommendations(request_obj)

        # Format response
        result = {
            "user_profile": response.user_profile,
            "recommendations": [
                {
                    "id": rec.id,
                    "name": rec.name,
                    "spirit_type": rec.spirit_type,
                    "price": rec.price,
                    "reasoning": rec.reasoning,
                    "similarity_score": rec.similarity_score,
                    "interesting_fact": rec.interesting_fact,
                }
                for rec in response.recommendations
            ],
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error generating recommendations for user {username}: {e}")
        return jsonify({"error": f"Failed to generate recommendations: {str(e)}"}), 500


# Example usage
if __name__ == "__main__":
    # Example 1: Load from API
    api = BaxusAPI()
    collection = api.get_user_collection("baxus")
    print(f"Loaded {len(collection)} items from API")

    # Example 2: Load from file
    file_collection = load_collection_from_file("sample-user-wishlist.json")
    print(f"Loaded {len(file_collection)} items from file")

    # Print sample item
    if file_collection:
        print("\nSample item:")
        print(json.dumps(file_collection[0], indent=2))

    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
