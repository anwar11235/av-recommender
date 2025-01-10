from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ..config.config import Config

class FeatureEngineer:
    """Feature engineering class for processing multiple data sources."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def engineer_user_features(self,
                             demographics: pd.DataFrame,
                             transactions: pd.DataFrame,
                             ga4_events: pd.DataFrame,
                             loyalty_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer user features from multiple data sources.
        
        Args:
            demographics: User demographic data
            transactions: Historical transaction data
            ga4_events: Google Analytics 4 event data
            loyalty_data: Loyalty program data
            
        Returns:
            DataFrame with engineered user features
        """
        features = pd.DataFrame()
        
        # Demographic features
        features = self._process_demographics(demographics)
        
        # Transaction-based features
        trans_features = self._process_transactions(transactions)
        features = features.join(trans_features, how='left')
        
        # Engagement features from GA4
        ga4_features = self._process_ga4_events(ga4_events)
        features = features.join(ga4_features, how='left')
        
        # Loyalty features
        loyalty_features = self._process_loyalty_data(loyalty_data)
        features = features.join(loyalty_features, how='left')
        
        # Fill missing values with appropriate defaults
        features = self._handle_missing_values(features)
        
        return features
    
    def engineer_offer_features(self,
                              mop_offers: pd.DataFrame,
                              wildfire_offers: pd.DataFrame,
                              historical_performance: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Engineer offer features from multiple sources.
        
        Args:
            mop_offers: MOP platform offer data
            wildfire_offers: Wildfire platform offer data
            historical_performance: Historical offer performance data (optional)
            
        Returns:
            DataFrame with engineered offer features
        """
        # Process MOP offers
        mop_features = self._process_mop_offers(mop_offers)
        
        # Process Wildfire offers
        wildfire_features = self._process_wildfire_offers(wildfire_offers)
        
        # Combine offer sources
        features = pd.concat([mop_features, wildfire_features], ignore_index=True)
        
        # Add historical performance metrics if available
        if historical_performance is not None:
            perf_features = self._process_historical_performance(historical_performance)
            features = features.join(perf_features, how='left')
        
        return features
    
    def _process_demographics(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Process demographic features."""
        features = pd.DataFrame(index=demographics.index)
        
        # Age-related features
        if 'birth_date' in demographics.columns:
            features['age'] = (datetime.now() - pd.to_datetime(demographics['birth_date'])).dt.years
            features['age_group'] = pd.cut(features['age'],
                                         bins=[0, 25, 35, 50, 65, 100],
                                         labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        # Location features
        if 'location' in demographics.columns:
            features['location'] = demographics['location'].fillna('UNKNOWN')
            # Create location embeddings or encodings
            self.label_encoders['location'] = LabelEncoder()
            features['location_encoded'] = self.label_encoders['location'].fit_transform(features['location'])
        
        # Gender encoding
        if 'gender' in demographics.columns:
            features['gender'] = demographics['gender'].fillna('UNKNOWN')
            self.label_encoders['gender'] = LabelEncoder()
            features['gender_encoded'] = self.label_encoders['gender'].fit_transform(features['gender'])
        
        return features
    
    def _process_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Process transaction history features."""
        features = pd.DataFrame(index=transactions['user_id'].unique())
        
        # Aggregate by user
        user_transactions = transactions.groupby('user_id')
        
        # Transaction frequency and recency
        features['transaction_count_30d'] = user_transactions.apply(
            lambda x: len(x[x['timestamp'] >= datetime.now() - timedelta(days=30)])
        )
        features['days_since_last_transaction'] = user_transactions['timestamp'].max().apply(
            lambda x: (datetime.now() - x).days
        )
        
        # Spending patterns
        features['average_transaction_amount'] = user_transactions['amount'].mean()
        features['total_spend_30d'] = user_transactions.apply(
            lambda x: x[x['timestamp'] >= datetime.now() - timedelta(days=30)]['amount'].sum()
        )
        
        # Category preferences
        category_counts = transactions.groupby(['user_id', 'category']).size().unstack(fill_value=0)
        category_preferences = category_counts.div(category_counts.sum(axis=1), axis=0)
        features = features.join(category_preferences.add_prefix('category_pref_'))
        
        return features
    
    def _process_ga4_events(self, ga4_events: pd.DataFrame) -> pd.DataFrame:
        """Process GA4 event features."""
        features = pd.DataFrame(index=ga4_events['user_id'].unique())
        
        # Engagement metrics
        user_events = ga4_events.groupby('user_id')
        
        features['page_views_30d'] = user_events.apply(
            lambda x: len(x[
                (x['event_name'] == 'page_view') & 
                (x['timestamp'] >= datetime.now() - timedelta(days=30))
            ])
        )
        
        features['offer_clicks_30d'] = user_events.apply(
            lambda x: len(x[
                (x['event_name'] == 'offer_click') & 
                (x['timestamp'] >= datetime.now() - timedelta(days=30))
            ])
        )
        
        # Session-based features
        features['avg_session_duration'] = user_events.apply(
            lambda x: x.groupby('session_id')['timestamp'].agg(lambda y: (y.max() - y.min()).total_seconds()).mean()
        )
        
        return features
    
    def _process_loyalty_data(self, loyalty_data: pd.DataFrame) -> pd.DataFrame:
        """Process loyalty program features."""
        features = pd.DataFrame(index=loyalty_data.index)
        
        # Tier-based features
        features['loyalty_tier'] = loyalty_data['tier']
        features['tier_weight'] = loyalty_data['tier'].map(Config.TIER_WEIGHTS)
        
        # Points and rewards
        features['current_points'] = loyalty_data['points_balance']
        features['lifetime_points'] = loyalty_data['lifetime_points']
        features['rewards_redeemed_30d'] = loyalty_data['rewards_redeemed_30d']
        
        # Engagement score
        features['loyalty_engagement_score'] = (
            (features['tier_weight'] * 0.3) +
            (features['rewards_redeemed_30d'] / features['rewards_redeemed_30d'].max() * 0.3) +
            (features['current_points'] / features['current_points'].max() * 0.4)
        )
        
        return features
    
    def _process_mop_offers(self, mop_offers: pd.DataFrame) -> pd.DataFrame:
        """Process MOP platform offer features."""
        features = pd.DataFrame(index=mop_offers.index)
        
        # Basic offer features
        features['source'] = 'MOP'
        features['priority_weight'] = Config.MOP_PRIORITY_WEIGHT
        
        # Offer characteristics
        for feature in Config.OFFER_FEATURES:
            if feature in mop_offers.columns:
                features[feature] = mop_offers[feature]
                
                # Encode categorical features
                if mop_offers[feature].dtype == 'object':
                    self.label_encoders[f'mop_{feature}'] = LabelEncoder()
                    features[f'{feature}_encoded'] = self.label_encoders[f'mop_{feature}'].fit_transform(features[feature])
        
        return features
    
    def _process_wildfire_offers(self, wildfire_offers: pd.DataFrame) -> pd.DataFrame:
        """Process Wildfire platform offer features."""
        features = pd.DataFrame(index=wildfire_offers.index)
        
        # Basic offer features
        features['source'] = 'WILDFIRE'
        features['priority_weight'] = Config.WILDFIRE_PRIORITY_WEIGHT
        
        # Offer characteristics
        for feature in Config.OFFER_FEATURES:
            if feature in wildfire_offers.columns:
                features[feature] = wildfire_offers[feature]
                
                # Encode categorical features
                if wildfire_offers[feature].dtype == 'object':
                    self.label_encoders[f'wildfire_{feature}'] = LabelEncoder()
                    features[f'{feature}_encoded'] = self.label_encoders[f'wildfire_{feature}'].fit_transform(features[feature])
        
        return features
    
    def _process_historical_performance(self, performance_data: pd.DataFrame) -> pd.DataFrame:
        """Process historical offer performance features."""
        features = pd.DataFrame(index=performance_data.index)
        
        # Performance metrics
        features['historical_ctr'] = performance_data['clicks'] / performance_data['impressions']
        features['historical_conversion_rate'] = performance_data['conversions'] / performance_data['clicks']
        features['avg_engagement_time'] = performance_data['total_engagement_time'] / performance_data['sessions']
        
        # Normalize performance metrics
        performance_cols = ['historical_ctr', 'historical_conversion_rate', 'avg_engagement_time']
        features[performance_cols] = self.scaler.fit_transform(features[performance_cols])
        
        return features
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Numeric columns: fill with median
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median())
        
        # Categorical columns: fill with mode
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            features[col] = features[col].fillna(features[col].mode()[0])
            
        return features 