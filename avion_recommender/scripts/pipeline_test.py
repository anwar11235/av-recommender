"""
# Avion Recommender System Test Pipeline

This notebook demonstrates the complete pipeline of the Avion Recommender System, including:
- Data generation
- Feature engineering
- Model training
- Recommendation testing
"""

"""
## Setup and Installation

Run these commands to set up the environment:
"""

# !git clone https://github.com/anwar11235/av-recommender.git
# %cd av-recommender
# !pip install -r requirements.txt

"""
## Import Libraries

Import required packages and custom modules:
"""

import sys
sys.path.append('/content/av-recommender')

from avion_recommender.data.synthetic_data_generator import SyntheticDataGenerator
from avion_recommender.utils.feature_engineering import FeatureEngineer
from avion_recommender.utils.feature_selector import FeatureSelector
from avion_recommender.models.base_recommender import BaseRecommender

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
## Generate Synthetic Data

Create synthetic data for testing the recommender system:
"""

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SyntheticDataGenerator(
        n_users=1000,
        n_offers_mop=100,
        n_offers_wildfire=100,
        n_interactions=10000
    )

    demographics, transactions, offers, interactions = generator.generate_all_data()

    print("Data generated successfully!")
    print(f"Users: {len(demographics)}")
    print(f"Transactions: {len(transactions)}")
    print(f"Offers: {len(offers)}")
    print(f"Interactions: {len(interactions)}")

    """
    ## Feature Engineering
    
    Engineer features for users and offers:
    """

    print("\nEngineering features...")
    feature_engineer = FeatureEngineer()

    user_features = feature_engineer.engineer_user_features(
        demographics=demographics,
        transactions=transactions,
        ga4_events=pd.DataFrame(),
        loyalty_data=demographics[['loyalty_tier']]
    )

    offer_features = feature_engineer.engineer_offer_features(
        mop_offers=offers[offers['source'] == 'MOP'],
        wildfire_offers=offers[offers['source'] == 'WILDFIRE']
    )

    print("Features engineered successfully!")
    print("\nUser features:")
    print(user_features.head())
    print("\nOffer features:")
    print(offer_features.head())

    """
    ## Train Recommender Model
    
    Create interaction matrix and train the recommender model:
    """

    print("\nTraining recommender model...")
    recommender = BaseRecommender()

    # Create interaction matrix
    user_ids = list(demographics.index)
    offer_ids = list(offers.index)
    interaction_matrix = np.zeros((len(user_ids), len(offer_ids)))

    for _, interaction in interactions.iterrows():
        user_idx = user_ids.index(interaction['user_id'])
        offer_idx = offer_ids.index(interaction['offer_id'])
        interaction_matrix[user_idx, offer_idx] += 1

    # Fit model
    recommender.fit(
        user_features=user_features,
        offer_features=offer_features,
        interaction_matrix=interaction_matrix
    )

    print("Model trained successfully!")

    """
    ## Test Recommendations
    
    Test both regular and cold start recommendations:
    """

    print("\nTesting recommendations...")
    test_user_id = user_ids[0]

    # Regular recommendations
    recommendations = recommender.get_recommendations(
        user_id=test_user_id,
        n_recommendations=5
    )
    print("Regular recommendations:")
    print(recommendations)

    # Cold start recommendations
    new_user = {
        'user_id': 'NEW_USER',
        'age_group': '26-35',
        'location': 'NYC',
        'gender': 'M'
    }
    cold_start_recs = recommender.handle_cold_start(new_user)
    print("\nCold start recommendations:")
    print(cold_start_recs)

if __name__ == "__main__":
    main() 