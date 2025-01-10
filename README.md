# Avion Recommender System

A personalized offer recommendation engine for rewards and loyalty program members.

## Features
- Personalized offer recommendations based on multiple data points
- Cold start handling for new users
- Real-time session-aware recommendations
- Tier-based filtering
- Multi-source content integration (MOP and Wildfire)
- Analytics and reporting

## Project Structure
```
avion_recommender/
├── data/                    # Data storage
├── models/                  # Model implementations
├── config/                  # Configuration files
├── api/                     # API endpoints
├── utils/                   # Utility functions
├── notebooks/              # Jupyter notebooks for analysis
└── tests/                  # Unit tests
```

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
- Copy `.env.example` to `.env`
- Fill in required API keys and configurations

## Development
- Use `jupyter notebook` to run analysis notebooks
- Run tests with `pytest`
- Format code with `black`
- Check code style with `flake8`

## Model Training
Instructions for model training will be added as components are developed.

## API Documentation
API documentation will be available at `/docs` when the service is running. 