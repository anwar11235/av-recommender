from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from ..config.config import Config

class DataProcessor:
    """Utility class for data processing and feature engineering."""
    
    @staticmethod
    def process_user_data(raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw user data into feature-engineered format.
        
        Args:
            raw_data: Raw user data DataFrame
            
        Returns:
            Processed DataFrame with engineered features
        """
        df = raw_data.copy()
        
        # Process demographic features
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 50, 65, 100],
                                   labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        # Process location data
        if 'location' in df.columns:
            df['location'] = df['location'].fillna('UNKNOWN')
            
        # Process transaction history
        if 'transactions' in df.columns:
            df['transaction_frequency'] = df['transactions'].apply(len)
            df['average_spend'] = df['transactions'].apply(
                lambda x: np.mean([t['amount'] for t in x]) if x else 0
            )
            
        return df[Config.USER_FEATURES]
    
    @staticmethod
    def process_offer_data(mop_data: pd.DataFrame, 
                          wildfire_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and merge offer data from multiple sources.
        
        Args:
            mop_data: MOP platform offer data
            wildfire_data: Wildfire platform offer data
            
        Returns:
            Processed and merged offer DataFrame
        """
        # Process MOP data
        mop_processed = mop_data.copy()
        mop_processed['source'] = 'MOP'
        mop_processed['priority_weight'] = Config.MOP_PRIORITY_WEIGHT
        
        # Process Wildfire data
        wildfire_processed = wildfire_data.copy()
        wildfire_processed['source'] = 'WILDFIRE'
        wildfire_processed['priority_weight'] = Config.WILDFIRE_PRIORITY_WEIGHT
        
        # Merge and standardize
        combined = pd.concat([mop_processed, wildfire_processed], ignore_index=True)
        
        return combined[Config.OFFER_FEATURES + ['source', 'priority_weight']]
    
    @staticmethod
    def create_interaction_matrix(user_ids: List[str], 
                                offer_ids: List[str], 
                                interactions: List[Dict]) -> np.ndarray:
        """
        Create user-offer interaction matrix.
        
        Args:
            user_ids: List of user IDs
            offer_ids: List of offer IDs
            interactions: List of interaction events
            
        Returns:
            Interaction matrix as numpy array
        """
        user_idx = {uid: i for i, uid in enumerate(user_ids)}
        offer_idx = {oid: i for i, oid in enumerate(offer_ids)}
        
        matrix = np.zeros((len(user_ids), len(offer_ids)))
        
        for interaction in interactions:
            user_id = interaction['user_id']
            offer_id = interaction['offer_id']
            interaction_type = interaction['type']
            
            if user_id in user_idx and offer_id in offer_idx:
                i, j = user_idx[user_id], offer_idx[offer_id]
                # Weight different types of interactions
                weight = 1.0 if interaction_type == 'conversion' else 0.5
                matrix[i, j] += weight
                
        return matrix 