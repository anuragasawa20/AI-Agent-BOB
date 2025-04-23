import os
from flask import Flask, request, jsonify
from bob_ai.main import (
    WhiskyRecommendationSystem,
    BaxusAPI,
    load_collection_from_file,
)
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Path to pre-computed embeddings (stored in the repo)
PRECOMPUTED_EMBEDDINGS = "./bob_ai/whisky_embeddings"

# Where embeddings will be used in the app
PERSIST_DIR = os.environ.get("PERSIST_DIR", "./whisky_embeddings")
DATA_PATH = os.environ.get("DATA_PATH", "./501 Bottle Dataset.csv")
PORT = int(os.environ.get("PORT", 5001))

# Initialize the recommendation system
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    )
    raise ValueError("OpenAI API key not found")


# Function to initialize the recommendation system
def initialize_system():
    logger.info(
        f"Setting up WhiskyRecommendationSystem with persist_directory={PERSIST_DIR}"
    )
    try:
        # Create directory if it doesn't exist
        Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

        # Check if embeddings directory is empty
        if not os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
            # Copy pre-computed embeddings if available
            if os.path.exists(PRECOMPUTED_EMBEDDINGS):
                logger.info(
                    f"Copying pre-computed embeddings from {PRECOMPUTED_EMBEDDINGS}"
                )

                # List files in precomputed directory to verify
                embedding_files = os.listdir(PRECOMPUTED_EMBEDDINGS)
                logger.info(f"Found embedding files: {embedding_files}")

                # Copy all files from precomputed dir to persist dir
                for item in embedding_files:
                    source = os.path.join(PRECOMPUTED_EMBEDDINGS, item)
                    dest = os.path.join(PERSIST_DIR, item)

                    if os.path.isdir(source):
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source, dest)

                logger.info(f"Successfully copied embeddings to {PERSIST_DIR}")
            else:
                logger.warning(
                    f"No pre-computed embeddings found at {PRECOMPUTED_EMBEDDINGS}"
                )
                logger.warning(
                    "Embeddings will be generated on first use (may be slow)"
                )
        else:
            logger.info(f"Using existing embeddings in {PERSIST_DIR}")

        # Initialize the system with existing or copied embeddings
        system = WhiskyRecommendationSystem(
            data_path=DATA_PATH, openai_api_key=api_key, persist_directory=PERSIST_DIR
        )
        logger.info("Whisky Recommendation System initialized successfully")
        return system
    except Exception as e:
        logger.error(f"Failed to initialize Whisky Recommendation System: {e}")
        raise


# Initialize the system
recommendation_system = initialize_system()


@app.route("/", methods=["GET"])
def index():
    """Root endpoint."""
    return (
        jsonify(
            {
                "status": "online",
                "message": "Bob AI Whisky Recommendation System API",
                "endpoints": [
                    "/health",
                    "/api/recommendations",
                    "/api/analyze",
                    "/get/recommendation/<username>",
                ],
            }
        ),
        200,
    )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return (
        jsonify(
            {
                "status": "healthy",
                "message": "Bob AI Whisky Recommendation System is running",
                "embeddings_path": PERSIST_DIR,
                "precomputed_embeddings_path": PRECOMPUTED_EMBEDDINGS,
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

        if not collection and not wishlist:
            return (
                jsonify({"error": "Either collection or wishlist must be provided"}),
                400,
            )

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
        return jsonify({"error": str(e)}), 500


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


# Add an endpoint to check embeddings status
@app.route("/embeddings/status", methods=["GET"])
def embeddings_status():
    """Check if embeddings are properly loaded."""
    try:
        if (
            hasattr(recommendation_system, "vector_store")
            and recommendation_system.vector_store is not None
        ):
            count = recommendation_system.vector_store._collection.count()
            return (
                jsonify(
                    {
                        "status": "loaded",
                        "embeddings_count": count,
                        "persist_dir": PERSIST_DIR,
                        "precomputed_dir": PRECOMPUTED_EMBEDDINGS,
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {"status": "not_loaded", "error": "Vector store not initialized"}
                ),
                500,
            )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # Start server
    app.run(host="0.0.0.0", port=PORT, debug=False)
