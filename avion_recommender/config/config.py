from typing import Dict, List

class Config:
    """Configuration settings for the recommender system."""
    
    # Model parameters
    EMBEDDING_DIM = 64
    LEARNING_RATE = 0.001
    BATCH_SIZE = 256
    
    # Feature engineering
    USER_FEATURES = [
        'loyalty_tier',
        'location',
        'age_group',
        'transaction_frequency',
        'average_spend'
    ]
    
    OFFER_FEATURES = [
        'category',
        'merchant',
        'reward_type',
        'discount_value',
        'expiration_days'
    ]
    
    # Data source configurations
    MOP_PRIORITY_WEIGHT = 0.7
    WILDFIRE_PRIORITY_WEIGHT = 0.3
    
    # Session parameters
    SESSION_EXPIRY_MINUTES = 30
    MAX_SESSION_INTERACTIONS = 100
    
    # Loyalty tiers and their weights
    TIER_WEIGHTS: Dict[str, float] = {
        'platinum': 1.0,
        'gold': 0.8,
        'silver': 0.6,
        'bronze': 0.4
    }
    
    # Cold start parameters
    MIN_INTERACTIONS_THRESHOLD = 5
    POPULARITY_WEIGHT = 0.4
    DEMOGRAPHIC_WEIGHT = 0.6
    
    # Analytics tracking
    TRACKING_METRICS: List[str] = [
        'click_through_rate',
        'conversion_rate',
        'average_engagement_time',
        'unique_offers_viewed'
    ] 