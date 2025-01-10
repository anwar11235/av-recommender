from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .cold_start_handler import ColdStartHandler
from ..config.config import Config

class BaseRecommender:
    """Base recommender system implementing core functionality."""
    
    def __init__(self):
        self.user_features = None
        self.offer_features = None
        self.interaction_matrix = None
        self.scaler = StandardScaler()
        self.cold_start_handler = ColdStartHandler()
        self.known_users = set()
        self.known_offers = set()
        
    def fit(self, 
            user_features: pd.DataFrame,
            offer_features: pd.DataFrame,
            interaction_matrix: np.ndarray):
        """
        Fit the recommender model with training data.
        
        Args:
            user_features: DataFrame of user features
            offer_features: DataFrame of offer features
            interaction_matrix: User-offer interaction matrix
        """
        self.user_features = user_features
        self.offer_features = offer_features
        self.interaction_matrix = interaction_matrix
        
        # Store known users and offers
        self.known_users = set(user_features.index)
        self.known_offers = set(offer_features.index)
        
        # Fit cold start handler
        self.cold_start_handler.fit(
            user_features,
            offer_features,
            interaction_matrix
        )
        
    def preprocess_user_features(self, user_data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess user features including demographics, loyalty tier, and location.
        
        Args:
            user_data: DataFrame containing user information
            
        Returns:
            Preprocessed user features as numpy array
        """
        processed_features = self.scaler.fit_transform(user_data)
        return processed_features
        
    def preprocess_offer_features(self, offer_data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess offer features from multiple sources (MOP and Wildfire).
        
        Args:
            offer_data: DataFrame containing offer information
            
        Returns:
            Preprocessed offer features as numpy array
        """
        processed_features = self.scaler.fit_transform(offer_data)
        return processed_features
        
    def update_session_state(self, user_id: str, interactions: List[Dict]) -> None:
        """
        Update user session state based on real-time interactions.
        
        Args:
            user_id: Unique identifier for the user
            interactions: List of interaction events (clicks, views, etc.)
        """
        if not interactions:
            return
            
        # Update interaction matrix for known users
        if user_id in self.known_users:
            user_idx = list(self.known_users).index(user_id)
            for interaction in interactions:
                offer_id = interaction['offer_id']
                if offer_id in self.known_offers:
                    offer_idx = list(self.known_offers).index(offer_id)
                    interaction_type = interaction['type']
                    weight = 1.0 if interaction_type == 'conversion' else 0.5
                    self.interaction_matrix[user_idx, offer_idx] += weight
        
    def get_recommendations(self, 
                          user_id: str, 
                          n_recommendations: int = 5,
                          user_tier: Optional[str] = None) -> List[str]:
        """
        Get personalized offer recommendations for a user.
        
        Args:
            user_id: Unique identifier for the user
            n_recommendations: Number of recommendations to return
            user_tier: User's loyalty tier for filtering
            
        Returns:
            List of recommended offer IDs
        """
        # Check if this is a cold start scenario
        if user_id not in self.known_users:
            # Get user data for cold start
            user_data = self._get_user_data(user_id)
            return self.cold_start_handler.get_recommendations_new_user(
                user_data,
                n_recommendations,
                user_tier
            )
            
        # Regular recommendation logic for known users
        user_idx = list(self.known_users).index(user_id)
        
        # Get user's interaction vector
        user_interactions = self.interaction_matrix[user_idx]
        
        # Calculate recommendation scores
        scores = self._calculate_recommendation_scores(user_idx, user_interactions)
        
        # Apply tier filtering if specified
        if user_tier:
            scores = self._filter_by_tier(scores, user_tier)
        
        # Get top N recommendations
        top_indices = np.argsort(scores)[-n_recommendations:][::-1]
        return [list(self.known_offers)[idx] for idx in top_indices]
    
    def handle_cold_start(self, user_data: Dict) -> List[str]:
        """
        Handle recommendations for new users with limited history.
        
        Args:
            user_data: Basic user information available
            
        Returns:
            List of recommended offer IDs
        """
        return self.cold_start_handler.get_recommendations_new_user(
            user_data,
            n_recommendations=5
        )
    
    def _calculate_recommendation_scores(self, 
                                      user_idx: int, 
                                      user_interactions: np.ndarray) -> np.ndarray:
        """Calculate recommendation scores for a user."""
        # This is a placeholder for the actual recommendation logic
        # In practice, this would use collaborative filtering, matrix factorization, etc.
        return user_interactions
    
    def _filter_by_tier(self, scores: np.ndarray, user_tier: str) -> np.ndarray:
        """Filter recommendations based on user tier."""
        tier_weight = Config.TIER_WEIGHTS.get(user_tier.lower(), 0.0)
        return scores * tier_weight
    
    def _get_user_data(self, user_id: str) -> Dict:
        """Get user data for cold start handling."""
        # This would typically fetch user data from your database
        # Placeholder implementation
        return {
            'user_id': user_id,
            'age_group': '26-35',  # Default values
            'location': 'UNKNOWN',
            'gender': 'O'
        } 