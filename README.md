# Bob AI Whisky Recommendation System

Bob is an AI agent built to analyze users' virtual whisky collections and provide personalized bottle recommendations. This project implements the RAG (Retrieval-Augmented Generation) component of Bob's recommendation engine.

## Overview

Bob AI connects to the Baxus service to analyze your existing whisky collection and provides personalized recommendations tailored to your taste profile. The system uses advanced natural language processing and semantic similarity to understand the nuances of whisky flavor profiles and match them to your preferences.

## Key Features

- **Collection Integration**: Seamlessly connects with your Baxus collection using just your username
- **Personalized Recommendations**: Analyzes your collection to suggest new whiskies you'll love
- **Detailed Reasoning**: Provides specific reasoning for each recommendation based on your preferences
- **Interesting Whisky Facts**: Includes fascinating facts about each recommended whisky
- **Price Awareness**: Filters recommendations by price range to match your budget
- **Spirit Type Analysis**: Identifies patterns in your preferred spirit categories
- **Characteristic Detection**: Recognizes your preference for characteristics like high-proof or specific price tiers

## API Endpoints

- **GET /get/recommendation/{username}**: Get recommendations using your Baxus username
- **POST /api/recommendations**: Get recommendations by providing your collection
- **POST /api/analyze**: Analyze your collection to understand your preference profile


## Architecture

The system follows a RAG-based approach as illustrated in the architectural diagram:

1. **Data Sources**:
   - Whisky catalog data (500 bottles)
   - User collection data

2. **Vector Database**:
   - Whisky embeddings
   - Flavor profile vectors
   - Metadata (region, price, age)

3. **Vector Search**:
   - Semantic similarity matching

4. **LLM Recommendation Engine**:
   - Context window with user profile and similar whisky data
   - Generation of ranked recommendations with personalized reasoning

## Installation

### Prerequisites
- Python 3.8+
- An OpenAI API key

### Setup
1. Clone the repository:
   ```
   git clone <repository-url>
   cd bob-ai
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

### Running the Demo
To run the demo with sample data:

```
python demo.py
```

### Starting the API Server
To start the Flask API server:

```
python api.py
```

The server will run on http://localhost:5000 by default.

### API Endpoints

1. **Health Check**:
   ```
   GET /health
   ```

2. **Get Recommendations**:
   ```
   POST /api/recommendations
   ```
   Request body:
   ```json
   {
     "collection": [
       {
         "id": 123,
         "name": "Whisky Name",
         "spirit_type": "Bourbon",
         "avg_msrp": 50.0,
         "abv": 45.0
       },
       ...
     ],
     "max_recommendations": 5
   }
   ```

3. **Analyze Collection**:
   ```
   POST /api/analyze
   ```
   Request body:
   ```json
   {
     "collection": [
       {
         "id": 123,
         "name": "Whisky Name",
         "spirit_type": "Bourbon",
         "avg_msrp": 50.0,
         "abv": 45.0
       },
       ...
     ]
   }
   ```

## Implementation Details

The system uses a combination of:
- Semantic embeddings with SentenceTransformers and OpenAI
- Vector search with Chroma
- LLM-based reasoning with OpenAI GPT-4
- Data analysis with Pandas and NumPy


## Technical Details

For developers and technical users interested in how Bob AI works under the hood, please see our [Technical Documentation](docs/technical_readme.md). 