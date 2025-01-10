from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from ..config.config import Config

class ColdStartHandler:
    """Handles cold start scenarios for new users and offers."""
    
    def __init__(self):
        self.demographic_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.offer_similarity_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.popular_offers = None
        self.demographic_features = None
        self.offer_features = None
        
    def fit(self, 
            user_features: pd.DataFrame,
            offer_features: pd.DataFrame,
            interaction_matrix: np.ndarray):
        """
        Fit the cold start models using existing data.
        
        Args:
            user_features: DataFrame of user features
            offer_features: DataFrame of offer features
            interaction_matrix: User-offer interaction matrix
        """
        # Prepare demographic features for similarity matching
        demographic_cols = ['age_group_encoded', 'location_encoded', 'gender_encoded']
        self.demographic_features = user_features[demographic_cols].values
        self.demographic_model.fit(self.demographic_features)
        
        # Prepare offer features for similarity matching
        offer_cols = [col for col in offer_features.columns if '_encoded' in col]
        self.offer_features = offer_features[offer_cols].values
        self.offer_similarity_model.fit(self.offer_features)
        
        # Calculate offer popularity
        self.popular_offers = self._calculate_offer_popularity(interaction_matrix)
        
    def get_recommendations_new_user(self,
                                   user_data: Dict,
                                   n_recommendations: int = 5,
                                   user_tier: Optional[str] = None) -> List[str]:
        """
        Get recommendations for a new user based on demographics and popular items.
        
        Args:
            user_data: Dictionary containing user demographic information
            n_recommendations: Number of recommendations to return
            user_tier: User's loyalty tier for filtering
            
        Returns:
            List of recommended offer IDs
        """
        # Process user demographics
        user_features = self._process_user_demographics(user_data)
        
        # Find similar users
        similar_user_indices = self._find_similar_users(user_features)
        
        # Get recommendations based on similar users and popularity
        recommendations = self._blend_recommendations(
            similar_user_indices,
            n_recommendations,
            user_tier
        )
        
        return recommendations
    
    def get_recommendations_new_offer(self,
                                    offer_data: Dict,
                                    n_similar_offers: int = 5) -> List[str]:
        """
        Find similar existing offers for a new offer.
        
        Args:
            offer_data: Dictionary containing offer information
            n_similar_offers: Number of similar offers to find
            
        Returns:
            List of similar offer IDs
        """
        # Process offer features
        offer_features = self._process_offer_features(offer_data)
        
        # Find similar offers
        distances, indices = self.offer_similarity_model.kneighbors(
            offer_features.reshape(1, -1),
            n_neighbors=n_similar_offers
        )
        
        return indices[0].tolist()
    
    def _process_user_demographics(self, user_data: Dict) -> np.ndarray:
        """Process user demographic data for similarity matching."""
        # Extract and encode demographic features
        features = np.zeros(len(self.demographic_features[0]))
        
        # Map categorical values to encoded values (should match training encoding)
        # This is a simplified version - in production, you'd need proper encoding persistence
        if 'age_group' in user_data:
            features[0] = self._encode_age_group(user_data['age_group'])
        if 'location' in user_data:
            features[1] = self._encode_location(user_data['location'])
        if 'gender' in user_data:
            features[2] = self._encode_gender(user_data['gender'])
            
        return features
    
    def _process_offer_features(self, offer_data: Dict) -> np.ndarray:
        """Process offer features for similarity matching."""
        # Extract and encode offer features
        features = np.zeros(len(self.offer_features[0]))
        
        # Map offer features to encoded values
        # This is a simplified version - in production, you'd need proper encoding persistence
        feature_mapping = {
            'category': 0,
            'merchant': 1,
            'reward_type': 2,
            'discount_value': 3
        }
        
        for feature, index in feature_mapping.items():
            if feature in offer_data:
                features[index] = self._encode_offer_feature(feature, offer_data[feature])
                
        return features
    
    def _find_similar_users(self, user_features: np.ndarray) -> np.ndarray:
        """Find similar users based on demographic features."""
        distances, indices = self.demographic_model.kneighbors(
            user_features.reshape(1, -1),
            n_neighbors=5
        )
        return indices[0]
    
    def _blend_recommendations(self,
                             similar_user_indices: np.ndarray,
                             n_recommendations: int,
                             user_tier: Optional[str]) -> List[str]:
        """Blend recommendations from similar users and popular offers."""
        # Get popular offers among similar users
        similar_user_offers = self._get_similar_user_offers(similar_user_indices)
        
        # Blend with overall popular offers
        blended_scores = {}
        
        # Weight from similar users
        for offer_id, score in similar_user_offers.items():
            blended_scores[offer_id] = score * Config.DEMOGRAPHIC_WEIGHT
            
        # Weight from popularity
        for offer_id, score in self.popular_offers.items():
            if offer_id in blended_scores:
                blended_scores[offer_id] += score * Config.POPULARITY_WEIGHT
            else:
                blended_scores[offer_id] = score * Config.POPULARITY_WEIGHT
        
        # Filter by tier if specified
        if user_tier:
            blended_scores = self._filter_by_tier(blended_scores, user_tier)
        
        # Sort and return top N recommendations
        sorted_offers = sorted(blended_scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        return [offer_id for offer_id, _ in sorted_offers[:n_recommendations]]
    
    def _calculate_offer_popularity(self, interaction_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate offer popularity scores."""
        # Sum interactions per offer
        offer_interactions = interaction_matrix.sum(axis=0)
        
        # Normalize scores
        normalized_scores = offer_interactions / offer_interactions.max()
        
        return dict(enumerate(normalized_scores))
    
    def _get_similar_user_offers(self, similar_user_indices: np.ndarray) -> Dict[str, float]:
        """Get weighted offer preferences from similar users."""
        # This would be implemented based on your interaction data structure
        # Placeholder implementation
        return {}
    
    def _filter_by_tier(self, offer_scores: Dict[str, float], user_tier: str) -> Dict[str, float]:
        """Filter offers based on user tier."""
        tier_weight = Config.TIER_WEIGHTS.get(user_tier.lower(), 0.0)
        return {k: v * tier_weight for k, v in offer_scores.items()}
    
    # Placeholder encoding methods - in production, these would use persistent encoders
    def _encode_age_group(self, age_group: str) -> int:
        age_groups = ['18-25', '26-35', '36-50', '51-65', '65+']
        return age_groups.index(age_group) if age_group in age_groups else 0
    
    def _encode_location(self, location: str) -> int:
        # Simplified encoding - would need proper mapping in production
        return hash(location) % 100
    
    def _encode_gender(self, gender: str) -> int:
        genders = ['M', 'F', 'O']
        return genders.index(gender) if gender in genders else 0
    
    def _encode_offer_feature(self, feature: str, value: str) -> int:
        # Simplified encoding - would need proper mapping in production
        return hash(f"{feature}_{value}") % 100 