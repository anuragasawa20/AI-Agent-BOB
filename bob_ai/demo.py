import os
import json
from main import WhiskyRecommendationSystem
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


def demo_whisky_recommendations():
    """
    Demonstrate the WhiskyRecommendationSystem with sample data.
    """
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )
        return

    print("Initializing Bob AI - Whisky Recommendation System...")

    # Sample user collection - we'll use a mix of different whisky types
    user_collection = [
        {
            "id": 164,
            "name": "Blanton's Original Single Barrel",
            "size": 750,
            "proof": 93,
            "abv": 46.5,
            "spirit_type": "Bourbon",
            "avg_msrp": 74.99,
            "fair_price": 104.52,
            "shelf_price": 139.86,
        },
        {
            "id": 466,
            "name": "Buffalo Trace",
            "size": 750,
            "proof": 90,
            "abv": 45,
            "spirit_type": "Bourbon",
            "avg_msrp": 26.99,
            "fair_price": 41.82,
            "shelf_price": 36.96,
        },
        {
            "id": 1663,
            "name": "Redbreast 12 Year Old Single Pot Still",
            "size": 750,
            "proof": 80,
            "abv": 40,
            "spirit_type": "Irish Whiskey",
            "avg_msrp": 66.08,
            "fair_price": 80.59,
            "shelf_price": 71.77,
        },
    ]

    # Initialize the recommendation system
    try:
        recommendation_system = WhiskyRecommendationSystem(
            data_path="../501 Bottle Dataset.csv", openai_api_key=api_key
        )

        print("System initialized successfully!")

        # Create a recommendation request
        request = WhiskyRecommendationSystem.RecommendationRequest(
            collection=user_collection, max_recommendations=5
        )

        print("\nAnalyzing user's collection...")
        print("User's collection:")
        for item in user_collection:
            print(f"- {item['name']} ({item['spirit_type']}): ${item['avg_msrp']}")

        print("\nGenerating recommendations...")
        # Get recommendations
        response = recommendation_system.get_recommendations(request)

        # Print user profile
        print("\n=== USER PROFILE ===")
        print(
            f"Preferred spirit types: {', '.join(response.user_profile['preferred_types']) if response.user_profile['preferred_types'] else 'No clear preference'}"
        )
        print(
            f"Price range: ${response.user_profile['price_range']['min']:.2f} - ${response.user_profile['price_range']['max']:.2f} (avg: ${response.user_profile['price_range']['avg']:.2f})"
        )
        print(
            f"ABV preference: {response.user_profile['abv_preference']['min']}% - {response.user_profile['abv_preference']['max']}% (avg: {response.user_profile['abv_preference']['avg']:.1f}%)"
        )

        # Print recommendations
        print("\n=== BOB'S RECOMMENDATIONS ===")
        for i, rec in enumerate(response.recommendations, 1):
            print(f"\n{i}. {rec.name} ({rec.spirit_type}) - ${rec.price:.2f}")
            print(f"   Match score: {rec.similarity_score:.2f}")
            print(f"   Reasoning: {rec.reasoning}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    demo_whisky_recommendations()
