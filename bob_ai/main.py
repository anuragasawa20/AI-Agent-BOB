import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, ForwardRef
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
import json
from collections import Counter
from scipy.spatial.distance import cosine
import openai
from dotenv import load_dotenv
import requests  # Add this import for the BaxusAPI class


# Add BaxusAPI class here
class BaxusAPI:
    """Client for interacting with the Baxus API."""

    def __init__(self, base_url: str = "http://services.baxus.co"):
        """Initialize the Baxus API client.

        Args:
            base_url: Base URL for the Baxus API
        """
        self.base_url = base_url

    def get_user_collection(self, username: str) -> List[Dict[str, Any]]:
        """Get a user's collection from the Baxus API.

        Args:
            username: Baxus username

        Returns:
            List of whisky objects in the user's collection
        """
        url = f"{self.base_url}/api/bar/user/{username}"
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return self._format_collection(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching user collection: {e}")
            return []

    def _format_collection(
        self, api_response: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format the API response into the required collection format.

        Args:
            api_response: Raw API response from Baxus

        Returns:
            Formatted collection for use with the recommendation system
        """
        collection = []

        for item in api_response:
            if "product" not in item:
                continue

            product = item["product"]

            # Map the product fields to our collection format
            whisky_item = {
                "id": product.get("id"),
                "name": product.get("name", "Unknown"),
                "size": product.get("size", 750),
                "proof": product.get("proof", 0),
                "abv": product.get("proof", 0)
                / 2,  # Estimate ABV from proof if not provided
                "spirit_type": product.get("spirit", "Unknown"),
                "avg_msrp": product.get("average_msrp", 0),
                "fair_price": product.get("fair_price", 0),
                "shelf_price": product.get("shelf_price", 0),
            }

            collection.append(whisky_item)

        return collection


def load_collection_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a collection from a local JSON file (for testing/development).

    Args:
        file_path: Path to the JSON file containing the collection

    Returns:
        Formatted collection for use with the recommendation system
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        api = BaxusAPI()
        return api._format_collection(data)
    except Exception as e:
        print(f"Error loading collection from file: {e}")
        return []


class WhiskyRecommendationSystem:
    """RAG-based system for personalized whisky recommendations."""

    # Define the class models at the beginning of the class
    class RecommendationRequest(BaseModel):
        """Request model for whisky recommendations."""

        collection: List[Dict[str, Any]] = Field(description="User's whisky collection")
        wishlist: List[Dict[str, Any]] = Field(
            default_factory=list, description="User's whisky wishlist"
        )
        username: Optional[str] = Field(
            default=None, description="Username to fetch collection from API"
        )
        max_recommendations: int = Field(
            default=5, description="Maximum number of recommendations to return"
        )

    class Recommendation(BaseModel):
        """Model for a single whisky recommendation."""

        id: Any = Field(description="Whisky ID")
        name: str = Field(description="Whisky name")
        spirit_type: str = Field(description="Spirit type")
        price: float = Field(description="Average MSRP")
        reasoning: str = Field(description="Reasoning for recommendation")
        similarity_score: float = Field(description="Similarity score")
        interesting_fact: str = Field(
            description="Interesting fact about the whisky or spirit type"
        )

    class RecommendationResponse(BaseModel):
        """Response model for whisky recommendations."""

        recommendations: List["WhiskyRecommendationSystem.Recommendation"] = Field(
            description="List of recommended whiskies"
        )
        user_profile: Dict[str, Any] = Field(
            description="User profile extracted from collection"
        )

    def __init__(
        self,
        data_path: str = "../501 Bottle Dataset.csv",
        openai_api_key: Optional[str] = None,
        persist_directory: str = "./whisky_embeddings",
    ):
        """
        Initialize the recommendation system.

        Args:
            data_path: Path to the CSV file containing whisky data.
            openai_api_key: OpenAI API key for embeddings and LLM.
            persist_directory: Directory to persist vector embeddings.
        """
        # Set API key
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Either pass it as a parameter or set it as an environment variable."
            )

        print("Whisky Recommendation System initializing...")

        # Load whisky data
        self.whisky_data = pd.read_csv(data_path)

        # Handle missing numeric values with 0 and string values with empty string
        numeric_cols = self.whisky_data.select_dtypes(include=["number"]).columns
        object_cols = self.whisky_data.select_dtypes(include=["object"]).columns

        for col in numeric_cols:
            self.whisky_data[col] = self.whisky_data[col].fillna(0)
        for col in object_cols:
            self.whisky_data[col] = self.whisky_data[col].fillna("")

        # Initialize embedding models
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Set persistence directory
        self.persist_directory = persist_directory

        # Create or load vector store
        self._create_or_load_vector_store()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4-1106-preview", temperature=0.7, api_key=self.api_key
        )

        # Initialize API client
        self.api = BaxusAPI()
        print("Whisky Recommendation System initialized successfully")

    def _create_or_load_vector_store(self):
        """Create a new vector store or load an existing one if available."""
        if (
            os.path.exists(self.persist_directory)
            and len(os.listdir(self.persist_directory)) > 0
        ):
            # Load existing vector store
            print("Loading existing vector embeddings...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="whisky_collection",
            )
            print(f"Loaded {self.vector_store._collection.count()} whisky embeddings")
        else:
            # Create new vector store
            print("Creating new vector embeddings...")
            self._create_vector_store()

    def _create_vector_store(self):
        """Create a vector store from the whisky data."""
        # Prepare documents for vector store
        documents = []
        for _, row in self.whisky_data.iterrows():
            # Create a rich description of each whisky
            content = f"""
            Name: {row['name']}
            Type: {row['spirit_type']}
            ABV: {row['abv']}
            Proof: {row['proof']}
            Average MSRP: ${row['avg_msrp']}
            Fair Price: ${row['fair_price']}
            Shelf Price: ${row['shelf_price']}
            Total Score: {row['total_score']}
            Ranking: {row['ranking']}
            Wishlist Count: {row['wishlist_count']}
            Vote Count: {row['vote_count']}
            Bar Count: {row['bar_count']}
            """

            # Store metadata for retrieval
            metadata = {
                "id": row["id"],
                "name": row["name"],
                "spirit_type": row["spirit_type"],
                "abv": row["abv"],
                "price": row["avg_msrp"],
                "fair_price": row["fair_price"],
                "shelf_price": row["shelf_price"],
                "ranking": row["ranking"],
                "popularity": row["popularity"],
                "total_score": row["total_score"],
                "wishlist_count": row["wishlist_count"],
                "vote_count": row["vote_count"],
                "bar_count": row["bar_count"],
            }

            documents.append(Document(page_content=content, metadata=metadata))

        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="whisky_collection",
            persist_directory=self.persist_directory,
        )

        print(f"Created and persisted {len(documents)} whisky embeddings")

    def _extract_user_preferences(
        self, user_collection: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract user preferences from their collection with detailed analysis.

        This enhanced version analyzes all spirit types and creates detailed profiles
        for each type based on price, ABV, and other characteristics.

        Args:
            user_collection: List of whisky objects in the user's collection.

        Returns:
            Dictionary containing detailed user preferences.
        """
        if not user_collection:
            return {
                "spirit_type_profiles": {},
                "overall_price_range": {"min": 0, "max": 0, "avg": 0},
                "overall_abv_preference": {"min": 0, "max": 0, "avg": 0},
                "type_frequency": {},
                "most_preferred_types": [],
            }

        # Extract collection data
        collection_df = pd.DataFrame(user_collection)

        # Process all spirit types (not just ones with 2+ occurrences)
        spirit_types = collection_df["spirit_type"].value_counts().to_dict()
        type_frequency = {k: v / len(collection_df) for k, v in spirit_types.items()}

        # Determine most preferred types (those above average frequency)
        avg_frequency = 1 / len(spirit_types) if spirit_types else 0
        most_preferred_types = [
            k for k, v in type_frequency.items() if v >= avg_frequency
        ]

        # Extract overall price preferences
        prices = collection_df["avg_msrp"].dropna().astype(float).tolist()
        overall_price_range = {
            "min": min(prices) if prices else 0,
            "max": max(prices) if prices else 0,
            "avg": sum(prices) / len(prices) if prices else 0,
        }

        # Extract overall ABV preferences
        abvs = collection_df["abv"].dropna().astype(float).tolist()
        overall_abv_preference = {
            "min": min(abvs) if abvs else 0,
            "max": max(abvs) if abvs else 0,
            "avg": sum(abvs) / len(abvs) if abvs else 0,
        }

        # Create detailed profiles for each spirit type
        spirit_type_profiles = {}
        for spirit_type in spirit_types.keys():
            # Filter for this spirit type
            type_df = collection_df[collection_df["spirit_type"] == spirit_type]

            # Extract type-specific price preferences
            type_prices = type_df["avg_msrp"].dropna().astype(float).tolist()
            price_range = {
                "min": min(type_prices) if type_prices else 0,
                "max": max(type_prices) if type_prices else 0,
                "avg": sum(type_prices) / len(type_prices) if type_prices else 0,
            }

            # Extract type-specific ABV preferences
            type_abvs = type_df["abv"].dropna().astype(float).tolist()
            abv_preference = {
                "min": min(type_abvs) if type_abvs else 0,
                "max": max(type_abvs) if type_abvs else 0,
                "avg": sum(type_abvs) / len(type_abvs) if type_abvs else 0,
            }

            # Analyze special characteristics of this type collection
            characteristics = self._analyze_type_characteristics(type_df)

            spirit_type_profiles[spirit_type] = {
                "count": spirit_types[spirit_type],
                "frequency": type_frequency[spirit_type],
                "price_range": price_range,
                "abv_preference": abv_preference,
                "characteristics": characteristics,
            }

        return {
            "spirit_type_profiles": spirit_type_profiles,
            "overall_price_range": overall_price_range,
            "overall_abv_preference": overall_abv_preference,
            "type_frequency": type_frequency,
            "most_preferred_types": most_preferred_types,
        }

    def _analyze_type_characteristics(self, type_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze specific characteristics that might explain user preference for this spirit type.

        Args:
            type_df: DataFrame containing the collection filtered for a specific spirit type

        Returns:
            Dictionary of characteristics that might explain preference
        """
        characteristics = {}

        # Check if high proof/ABV is a preference
        if not type_df.empty and "abv" in type_df.columns:
            avg_abv = type_df["abv"].dropna().mean()
            if avg_abv > 50:
                characteristics["high_proof"] = True
                characteristics["high_proof_level"] = (
                    "very high" if avg_abv > 55 else "high"
                )
            elif avg_abv < 40:
                characteristics["low_proof"] = True

        # Check if price range indicates a preference
        if not type_df.empty and "avg_msrp" in type_df.columns:
            avg_price = type_df["avg_msrp"].dropna().mean()
            if avg_price > 100:
                characteristics["premium_price"] = True
                characteristics["price_tier"] = (
                    "luxury" if avg_price > 200 else "premium"
                )
            elif avg_price < 30:
                characteristics["budget_price"] = True
                characteristics["price_tier"] = "budget"
            else:
                characteristics["price_tier"] = "mid-range"

        # Add any other characteristics that might explain preference
        # (could be extended with more sophisticated analysis)

        return characteristics

    def _generate_user_profile(self, user_collection: List[Dict[str, Any]]) -> str:
        """
        Generate a detailed text description of the user's preferences.

        This enhanced version includes detailed profiles for each spirit type
        and reasons why the user might prefer certain types.

        Args:
            user_collection: List of whisky objects in the user's collection.

        Returns:
            Detailed text description of user preferences.
        """
        preferences = self._extract_user_preferences(user_collection)

        # Start with overall profile
        profile = "User Profile:\n"

        # Add overall price and ABV preferences
        price_range = preferences["overall_price_range"]
        abv_preference = preferences["overall_abv_preference"]

        profile += f"- Overall price range: ${price_range['min']:.2f} - ${price_range['max']:.2f} (avg: ${price_range['avg']:.2f})\n"
        profile += f"- Overall ABV preference: {abv_preference['min']}% - {abv_preference['max']}% (avg: {abv_preference['avg']:.1f}%)\n"

        # Add most preferred types
        most_preferred = preferences["most_preferred_types"]
        profile += f"- Most preferred spirit types: {', '.join(most_preferred) if most_preferred else 'No clear preference'}\n"

        # Add detailed profiles for each spirit type
        profile += "\nDetailed Spirit Type Profiles:\n"

        for spirit_type, type_profile in preferences["spirit_type_profiles"].items():
            count = type_profile["count"]
            frequency = type_profile["frequency"] * 100

            profile += (
                f"\n{spirit_type} ({count} bottles, {frequency:.1f}% of collection):\n"
            )

            # Add type-specific price and ABV preferences
            price_range = type_profile["price_range"]
            abv_preference = type_profile["abv_preference"]

            profile += f"  - Price range: ${price_range['min']:.2f} - ${price_range['max']:.2f} (avg: ${price_range['avg']:.2f})\n"
            profile += f"  - ABV preference: {abv_preference['min']}% - {abv_preference['max']}% (avg: {abv_preference['avg']:.1f}%)\n"

            # Add specific characteristics for this type
            characteristics = type_profile["characteristics"]

            if characteristics:
                profile += "  - Notable characteristics:\n"

                for key, value in characteristics.items():
                    if key == "high_proof" and value:
                        profile += f"    * Preference for {characteristics.get('high_proof_level', 'high')} proof spirits\n"
                    elif key == "low_proof" and value:
                        profile += "    * Preference for lower proof spirits\n"
                    elif key == "premium_price" and value:
                        profile += f"    * Preference for {characteristics.get('price_tier', 'premium')} price tier\n"
                    elif key == "budget_price" and value:
                        profile += "    * Preference for budget-friendly options\n"
                    elif (
                        key == "price_tier"
                        and key != "premium_price"
                        and key != "budget_price"
                    ):
                        profile += f"    * {value.capitalize()} price tier preference\n"

        return profile

    def _semantic_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Perform semantic search on the whisky vector store.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            List of Document objects.
        """
        return self.vector_store.similarity_search(query, k=k)

    def _filter_recommendations(
        self,
        candidates: List[Document],
        user_collection: List[Dict[str, Any]],
        price_range_factor: float = 1.5,
    ) -> List[Document]:
        """
        Filter recommendations based on user preferences.

        Args:
            candidates: List of candidate Document objects.
            user_collection: List of whisky objects in the user's collection.
            price_range_factor: Factor to expand the price range.

        Returns:
            Filtered list of Document objects.
        """
        if not user_collection:
            return candidates

        preferences = self._extract_user_preferences(user_collection)

        # Get IDs of bottles already in collection
        collection_ids = [item.get("id") for item in user_collection]

        # Calculate expanded price range
        price_min = preferences["overall_price_range"]["min"] / price_range_factor
        price_max = preferences["overall_price_range"]["max"] * price_range_factor

        filtered = []
        for doc in candidates:
            # Skip if already in collection
            if doc.metadata.get("id") in collection_ids:
                continue

            # Filter by price if we have price preferences
            price = float(doc.metadata.get("price", 0))
            if preferences["overall_price_range"]["avg"] > 0 and (
                price < price_min or price > price_max
            ):
                continue

            filtered.append(doc)

        return filtered

    def _prioritize_recommendations(
        self, candidates: List[Document], user_preferences: Dict[str, Any]
    ) -> List[Document]:
        """
        Prioritize recommendations based on user's spirit type preferences while ensuring diversity.

        This function ensures:
        1. Maximum representation: A spirit type appears in recommendations at most according to its collection percentage
        2. Diversity: Never recommend only a single spirit type
        3. Empty collection handling: When no collection exists, show diverse recommendations based on popularity

        Args:
            candidates: List of filtered candidate Document objects.
            user_preferences: User preferences extracted from collection.

        Returns:
            Re-ordered list of Document objects prioritized by user preferences.
        """
        if not candidates:
            return []

        # Get type frequency from user preferences
        type_frequency = user_preferences.get("type_frequency", {})

        # Get list of spirit types in the user's collection
        collection_spirit_types = list(type_frequency.keys())

        # Define max recommendations to return
        max_recommendations = min(len(candidates), 5)  # Default to 5

        # Special case: Empty collection or very limited collection (1-2 items)
        if not collection_spirit_types or len(collection_spirit_types) < 2:
            return self._handle_empty_collection_recommendations(
                candidates, max_recommendations
            )

        # Define a scoring function for each candidate
        def score_candidate(doc):
            spirit_type = doc.metadata.get("spirit_type", "")

            # Base score is the frequency of this type in the user's collection
            # (default to 0 if type not in collection)
            type_score = type_frequency.get(spirit_type, 0)

            # Boost score based on whisky rating/popularity
            ranking = float(
                doc.metadata.get("ranking", 500)
            )  # Default to low if missing
            # Normalize ranking (lower is better) - use inverse and cap at 100
            ranking_score = (
                min(100 / max(ranking, 1), 1.0) * 0.3
            )  # 30% weight to ranking

            return type_score + ranking_score

        # Sort candidates by score (highest first)
        scored_candidates = [(score_candidate(doc), doc) for doc in candidates]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Determine maximum allocation per type based on collection percentages
        # and ensure no single type dominates
        type_allocations = {}

        # Calculate initial allocations based on frequencies (cap at their % representation)
        for spirit_type, freq in type_frequency.items():
            # Calculate the maximum number based on frequency (rounded up)
            max_of_type = min(
                max(1, round(freq * max_recommendations)),
                max_recommendations - 1,  # Ensure at least one slot for diversity
            )
            type_allocations[spirit_type] = max_of_type

        # Ensure we reserve at least one slot for a type not in the collection for diversity
        total_allocation = sum(type_allocations.values())
        if total_allocation >= max_recommendations:
            # Reduce allocations proportionally, starting with the most frequent types
            sorted_types = sorted(
                type_frequency.items(), key=lambda x: x[1], reverse=True
            )
            for spirit_type, _ in sorted_types:
                if total_allocation < max_recommendations:
                    break
                if type_allocations[spirit_type] > 1:  # Don't reduce to zero
                    type_allocations[spirit_type] -= 1
                    total_allocation -= 1

        # Allocate recommendations based on calculated distribution
        result = []
        type_counts = {t: 0 for t in type_allocations}
        remaining = []

        # First pass: Add recommendations within allocation limits
        for score, doc in scored_candidates:
            spirit_type = doc.metadata.get("spirit_type", "")

            if spirit_type in type_allocations and type_counts.get(
                spirit_type, 0
            ) < type_allocations.get(spirit_type, 0):
                result.append(doc)
                type_counts[spirit_type] = type_counts.get(spirit_type, 0) + 1
            else:
                remaining.append(doc)

            # Break if we have enough recommendations
            if len(result) >= max_recommendations:
                break

        # If we still have slots to fill, prioritize diversity
        remaining_slots = max_recommendations - len(result)
        if remaining_slots > 0:
            # Get all unique spirit types in remaining candidates
            remaining_types = set(
                doc.metadata.get("spirit_type", "") for _, doc in remaining
            )

            # Prioritize types not yet in the results
            represented_types = set(
                doc.metadata.get("spirit_type", "") for doc in result
            )
            diversity_candidates = [
                doc
                for _, doc in remaining
                if doc.metadata.get("spirit_type", "") not in represented_types
            ]

            # Add diverse recommendations first
            for doc in diversity_candidates:
                if remaining_slots <= 0:
                    break
                result.append(doc)
                remaining_slots -= 1
                represented_types.add(doc.metadata.get("spirit_type", ""))

            # Fill any remaining slots with highest scored remaining candidates
            if remaining_slots > 0:
                for _, doc in scored_candidates:
                    if doc not in result and remaining_slots > 0:
                        result.append(doc)
                        remaining_slots -= 1
                    if remaining_slots <= 0:
                        break

        return result

    def _handle_empty_collection_recommendations(
        self, candidates: List[Document], max_recommendations: int
    ) -> List[Document]:
        """
        Handle the case where user has no collection or very limited collection.
        Return diverse recommendations based on popularity and ensuring multiple spirit types.

        Args:
            candidates: List of candidate Document objects
            max_recommendations: Maximum number of recommendations to return

        Returns:
            List of diverse Document objects
        """
        # Group candidates by spirit type
        spirit_type_groups = {}
        for doc in candidates:
            spirit_type = doc.metadata.get("spirit_type", "")
            if spirit_type not in spirit_type_groups:
                spirit_type_groups[spirit_type] = []
            spirit_type_groups[spirit_type].append(doc)

        # Sort each group by popularity/ranking
        for spirit_type, docs in spirit_type_groups.items():
            docs.sort(key=lambda doc: float(doc.metadata.get("ranking", 500)))

        # Take top from each spirit type in rotation
        result = []
        spirit_types = list(spirit_type_groups.keys())

        if not spirit_types:
            return []

        # Ensure we have at least one from each available spirit type
        index = 0
        while len(result) < max_recommendations and len(result) < len(candidates):
            spirit_type = spirit_types[index % len(spirit_types)]
            if spirit_type_groups[spirit_type]:
                result.append(spirit_type_groups[spirit_type].pop(0))
            index += 1

            # Break if we've gone through all available candidates
            if all(len(group) == 0 for group in spirit_type_groups.values()):
                break

        return result

    def get_recommendations(
        self,
        request_or_query,
        username: Optional[str] = None,
        collection_file: Optional[str] = None,
        num_results: int = 3,
        price_range: Optional[tuple] = None,
        spirit_types: Optional[List[str]] = None,
    ) -> RecommendationResponse:
        """
        Get personalized whisky recommendations.

        This method supports multiple calling signatures:
        1. With a RecommendationRequest object: get_recommendations(request)
        2. With query parameters: get_recommendations(query, username, collection_file, etc.)

        Args:
            request_or_query: Either a RecommendationRequest object or a query string
            username: Baxus username (if provided, will fetch collection from API)
            collection_file: Path to collection file (alternative to username)
            num_results: Number of recommendations to return
            price_range: Optional tuple of (min_price, max_price)
            spirit_types: Optional list of spirit types to filter by

        Returns:
            RecommendationResponse containing recommendations, reasoning, and token usage stats
        """
        # Check if using the request object signature
        if isinstance(request_or_query, self.RecommendationRequest):
            request = request_or_query
            # If username is provided in the request and collection is empty, fetch from API
            if request.username and not request.collection:
                print(f"Fetching collection for user: {request.username}")
                collection = self.api.get_user_collection(request.username)
                if collection:
                    request.collection = collection
                else:
                    print(f"No collection found for user {request.username}")
            return self.get_recommendations_from_request(request)

        # Otherwise, use the query string signature
        query = request_or_query

        # Load the user's collection
        collection = []
        if username:
            print(f"Fetching collection for user: {username}")
            collection = self.api.get_user_collection(username)
        elif collection_file:
            print(f"Loading collection from file: {collection_file}")
            collection = load_collection_from_file(collection_file)

        if not collection:
            print("No collection found, using default whisky database")

        # Create a request object
        request = self.RecommendationRequest(
            collection=collection, max_recommendations=num_results
        )

        return self.get_recommendations_from_request(request)

    def get_recommendations_from_request(
        self, request: RecommendationRequest
    ) -> RecommendationResponse:
        """
        Get recommendations from a RecommendationRequest object.

        Args:
            request: RecommendationRequest object.

        Returns:
            RecommendationResponse object.
        """
        # Extract user preferences
        user_preferences = self._extract_user_preferences(request.collection)
        user_profile = self._generate_user_profile(request.collection)

        # Construct search query from user profile
        query = f"""
        Find whiskies similar to a collection with the following profile:
        {user_profile}
        """

        # Perform semantic search
        candidates = self._semantic_search(query, k=40)  # Increased to get more variety

        # Filter recommendations
        filtered_candidates = self._filter_recommendations(
            candidates, request.collection
        )

        # Prioritize recommendations based on user preferences
        prioritized_candidates = self._prioritize_recommendations(
            filtered_candidates, user_preferences
        )

        # Generate recommendations with reasoning using LLM
        recommendations = self._generate_recommendations_with_reasoning(
            prioritized_candidates,
            request.collection,
            user_preferences,
            max_recommendations=request.max_recommendations,
        )

        return self.RecommendationResponse(
            recommendations=recommendations,
            user_profile=user_preferences,
        )

    def _generate_recommendations_with_reasoning(
        self,
        candidates: List[Document],
        user_collection: List[Dict[str, Any]],
        user_preferences: Dict[str, Any],
        max_recommendations: int = 5,
    ) -> List[Recommendation]:
        """
        Generate recommendations with reasoning using LLM.
        Enhanced to use detailed user preferences for better reasoning and includes interesting facts.

        Args:
            candidates: List of filtered candidate Document objects.
            user_collection: List of whisky objects in the user's collection.
            user_preferences: Detailed user preferences from _extract_user_preferences
            max_recommendations: Maximum number of recommendations to return.

        Returns:
            List of Recommendation objects.
        """
        if not candidates:
            return []

        # Format user collection for prompt
        collection_text = ""
        for (
            item
        ) in user_collection:  # No need to limit to 10 items with only 3 in collection
            collection_text += f"- {item.get('name', 'Unknown')} ({item.get('spirit_type', 'Unknown')}): ${item.get('avg_msrp', 'Unknown')}, ABV: {item.get('abv', 'Unknown')}%\n"

        # Format detailed user profile for prompt
        user_profile_text = self._generate_user_profile(user_collection)

        # Create prompt for generating recommendations
        prompt_template = PromptTemplate.from_template(
            """
            You are Bob, a whisky expert AI assistant. You're helping a user find new whisky recommendations based on their collection.
            
            User's whisky collection:
            {collection}
            
            Detailed user profile:
            {user_profile}
            
            Potential recommendations:
            {candidates}
            
            Based on the detailed user profile, provide personalized reasoning for each recommendation. 
            Explain why they would like it based on their collection preferences, considering:
            1. Their spirit type preferences and characteristics they like about each type
            2. Their price range preferences
            3. Their ABV/proof preferences
            4. Any other notable patterns in their collection
            
            IMPORTANT: For each recommendation, include a fascinating and historically accurate fact about the spirit type name or its distillery.
            This fact should be interesting, and highlight something unique about the spirit type, its heritage, production, or flavor profile.
            
            IMPORTANT: Since the user has multiple spirit types in their collection (Bourbon, Rye, Irish Whiskey), 
            make sure to highlight how each recommendation relates to their diverse preferences.
            
            Your reasoning should be specific and reference the detailed profile insights.
            
            Return your recommendations as a JSON array of objects with the fields:
            - id: The whisky ID
            - name: The whisky name
            - spirit_type: The spirit type
            - price: The average MSRP as a float
            - reasoning: Your personalized reasoning based on detailed profile
            - similarity_score: A float from 0-1 indicating how well this matches their preferences
            - interesting_fact: A fascinating fact about the whisky or its distillery
            
            JSON:
            """
        )

        # Format candidates for prompt
        candidates_text = ""
        for i, doc in enumerate(candidates[:max_recommendations]):
            candidates_text += f"""
            Candidate {i+1}:
            ID: {doc.metadata.get('id', 'Unknown')}
            Name: {doc.metadata.get('name', 'Unknown')}
            Type: {doc.metadata.get('spirit_type', 'Unknown')}
            Price: ${doc.metadata.get('price', 'Unknown')}
            Fair Price: ${doc.metadata.get('fair_price', 'Unknown')}
            ABV: {doc.metadata.get('abv', 'Unknown')}%
            Ranking: {doc.metadata.get('ranking', 'Unknown')}
            Wishlist Count: {doc.metadata.get('wishlist_count', 'Unknown')}
            Vote Count: {doc.metadata.get('vote_count', 'Unknown')}
            Bar Count: {doc.metadata.get('bar_count', 'Unknown')}
            """

        # Generate recommendations with LLM
        prompt = prompt_template.format(
            collection=collection_text,
            user_profile=user_profile_text,
            candidates=candidates_text,
        )

        response = self.llm.invoke(prompt).content

        # Parse LLM response
        try:
            # Try to extract JSON from response
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()

            recommendations_data = json.loads(json_str)

            recommendations = []
            for rec_data in recommendations_data:
                recommendations.append(
                    self.Recommendation(
                        id=str(rec_data.get("id", "")),
                        name=rec_data.get("name", ""),
                        spirit_type=rec_data.get("spirit_type", ""),
                        price=float(rec_data.get("price", 0)),
                        reasoning=rec_data.get("reasoning", ""),
                        similarity_score=float(rec_data.get("similarity_score", 0)),
                        interesting_fact=rec_data.get(
                            "interesting_fact",
                            "Did you know? This whisky has a unique history worth exploring.",
                        ),
                    )
                )

            return recommendations

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response}")

            # Fallback: Return simple recommendations without reasoning
            return [
                self.Recommendation(
                    id=str(doc.metadata.get("id", "")),
                    name=doc.metadata.get("name", ""),
                    spirit_type=doc.metadata.get("spirit_type", ""),
                    price=float(doc.metadata.get("price", 0)),
                    reasoning="Based on your collection preferences.",
                    similarity_score=0.5,
                    interesting_fact="Did you know? Whisky has been produced for hundreds of years using traditional methods.",
                )
                for doc in candidates[:max_recommendations]
            ]


# Example mock data structure
mock_collection = [
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
        "id": 542,
        "name": "Eagle Rare 10 Year Old",
        "size": 750,
        "abv": 45,
        "spirit_type": "Bourbon",
        "avg_msrp": 39.9,
        "fair_price": 66.25,
        "shelf_price": 49.99,
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


# Helper function to print a recommendation response
def print_recommendation_response(response):
    """Helper function to print a recommendation response"""
    print("\nUser Profile:")
    # Print a few key insights from the detailed profile
    print(
        f"Most preferred types: {', '.join(response.user_profile['most_preferred_types'])}"
    )
    price_range = response.user_profile["overall_price_range"]
    print(
        f"Price range: ${price_range['min']:.2f} - ${price_range['max']:.2f} (avg: ${price_range['avg']:.2f})"
    )

    # Print type-specific insights
    print("\nSpirit Type Details:")
    for spirit_type, profile in response.user_profile["spirit_type_profiles"].items():
        print(
            f"- {spirit_type} ({profile['count']} bottles, {profile['frequency']*100:.1f}% of collection)"
        )

    print("\nRecommendations:")
    for i, rec in enumerate(response.recommendations, 1):
        print(f"\n{i}. {rec.name} ({rec.spirit_type}) - ${rec.price:.2f}")
        print(f"   Reasoning: {rec.reasoning}")
        print(f"   Interesting Fact: {rec.interesting_fact}")


# Example function to demonstrate usage
def example_usage():
    # Set your OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Warning: No API key found. Set the OPENAI_API_KEY environment variable.")
        return

    # Initialize the recommendation system
    recommendation_system = WhiskyRecommendationSystem(openai_api_key=api_key)

    # Example 1: Using mock collection directly
    print("\n=== Example 1: Using mock collection ===")
    request_with_collection = WhiskyRecommendationSystem.RecommendationRequest(
        collection=mock_collection, max_recommendations=3
    )
    response = recommendation_system.get_recommendations(request_with_collection)
    print_recommendation_response(response)

    # Example 2: Using username to fetch from API
    print("\n=== Example 2: Using username to fetch from API ===")
    request_with_username = WhiskyRecommendationSystem.RecommendationRequest(
        collection=[], username="baxus", max_recommendations=3
    )
    response = recommendation_system.get_recommendations(request_with_username)
    print_recommendation_response(response)

    # Example 3: Using the alternate method signature
    print("\n=== Example 3: Using alternate method signature ===")
    response = recommendation_system.get_recommendations(
        "Find me whiskies similar to my collection", username="baxus", num_results=3
    )
    print_recommendation_response(response)


if __name__ == "__main__":
    example_usage()
