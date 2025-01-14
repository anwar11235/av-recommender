import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import random
from ..config.config import Config

class SyntheticDataGenerator:
    """Generate synthetic data for testing the recommender system."""
    
    def __init__(self, 
                 n_users: int = 1000,
                 n_offers_mop: int = 100,
                 n_offers_wildfire: int = 100,
                 n_interactions: int = 10000):
        """
        Initialize the data generator.
        
        Args:
            n_users: Number of users to generate
            n_offers_mop: Number of MOP offers
            n_offers_wildfire: Number of Wildfire offers
            n_interactions: Number of user-offer interactions
        """
        self.n_users = n_users
        self.n_offers_mop = n_offers_mop
        self.n_offers_wildfire = n_offers_wildfire
        self.n_interactions = n_interactions
        
        # Constants for data generation
        self.locations = ['NYC', 'LA', 'CHI', 'HOU', 'PHX', 'PHL', 'SAN', 'DAL', 'MIA']
        self.categories = ['food', 'retail', 'travel', 'entertainment', 'services']
        self.merchants = [f'Merchant_{i}' for i in range(20)]
        self.reward_types = ['cashback', 'points', 'discount']
        self.event_types = ['view', 'click', 'conversion']
        
    def generate_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all necessary synthetic data.
        
        Returns:
            Tuple containing:
            - User demographics DataFrame
            - Transaction history DataFrame
            - Offer data DataFrame
            - Interaction data DataFrame
        """
        print("Generating user demographics...")
        demographics = self._generate_user_demographics()
        
        print("Generating transaction history...")
        transactions = self._generate_transactions(demographics)
        
        print("Generating offer data...")
        offers = self._generate_offers()
        
        print("Generating interaction data...")
        interactions = self._generate_interactions(demographics, offers)
        
        return demographics, transactions, offers, interactions
    
    def _generate_user_demographics(self) -> pd.DataFrame:
        """Generate synthetic user demographic data."""
        return pd.DataFrame({
            'user_id': [f'U{i:06d}' for i in range(self.n_users)],
            'age': np.random.randint(18, 80, self.n_users),
            'gender': np.random.choice(['M', 'F', 'O'], self.n_users),
            'location': np.random.choice(self.locations, self.n_users),
            'join_date': [
                datetime.now() - timedelta(days=np.random.randint(1, 1000))
                for _ in range(self.n_users)
            ],
            'loyalty_tier': np.random.choice(
                list(Config.TIER_WEIGHTS.keys()),
                self.n_users,
                p=[0.1, 0.2, 0.3, 0.4]  # Probability distribution for tiers
            )
        }).set_index('user_id')
    
    def _generate_transactions(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic transaction history."""
        n_transactions = self.n_users * 10  # Average 10 transactions per user
        
        transactions = pd.DataFrame({
            'transaction_id': [f'T{i:08d}' for i in range(n_transactions)],
            'user_id': np.random.choice(demographics.index, n_transactions),
            'timestamp': [
                datetime.now() - timedelta(days=np.random.randint(1, 365))
                for _ in range(n_transactions)
            ],
            'amount': np.random.normal(100, 30, n_transactions).clip(10, 500),
            'category': np.random.choice(self.categories, n_transactions),
            'merchant': np.random.choice(self.merchants, n_transactions)
        })
        
        return transactions.sort_values('timestamp')
    
    def _generate_offers(self) -> pd.DataFrame:
        """Generate synthetic offer data from both MOP and Wildfire."""
        # Generate MOP offers
        mop_offers = pd.DataFrame({
            'offer_id': [f'MOP{i:06d}' for i in range(self.n_offers_mop)],
            'source': 'MOP',
            'category': np.random.choice(self.categories, self.n_offers_mop),
            'merchant': np.random.choice(self.merchants, self.n_offers_mop),
            'reward_type': np.random.choice(self.reward_types, self.n_offers_mop),
            'discount_value': np.random.uniform(5, 50, self.n_offers_mop),
            'expiration_days': np.random.choice([7, 14, 30, 60], self.n_offers_mop),
            'min_spend': np.random.uniform(10, 100, self.n_offers_mop),
            'priority_weight': Config.MOP_PRIORITY_WEIGHT
        })
        
        # Generate Wildfire offers
        wildfire_offers = pd.DataFrame({
            'offer_id': [f'WF{i:06d}' for i in range(self.n_offers_wildfire)],
            'source': 'WILDFIRE',
            'category': np.random.choice(self.categories, self.n_offers_wildfire),
            'merchant': np.random.choice(self.merchants, self.n_offers_wildfire),
            'reward_type': np.random.choice(self.reward_types, self.n_offers_wildfire),
            'discount_value': np.random.uniform(5, 50, self.n_offers_wildfire),
            'expiration_days': np.random.choice([7, 14, 30, 60], self.n_offers_wildfire),
            'min_spend': np.random.uniform(10, 100, self.n_offers_wildfire),
            'priority_weight': Config.WILDFIRE_PRIORITY_WEIGHT
        })
        
        # Combine and return
        offers = pd.concat([mop_offers, wildfire_offers], ignore_index=True)
        return offers.set_index('offer_id')
    
    def _generate_interactions(self, 
                             demographics: pd.DataFrame, 
                             offers: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic user-offer interactions."""
        interactions = []
        
        for _ in range(self.n_interactions):
            # Select random user and offer
            user_id = np.random.choice(demographics.index)
            offer_id = np.random.choice(offers.index)
            
            # Generate interaction sequence (view -> click -> conversion)
            base_timestamp = datetime.now() - timedelta(days=np.random.randint(1, 30))
            
            # Always create a view
            interactions.append({
                'interaction_id': f'I{len(interactions):08d}',
                'user_id': user_id,
                'offer_id': offer_id,
                'type': 'view',
                'timestamp': base_timestamp
            })
            
            # 50% chance of click after view
            if random.random() < 0.5:
                interactions.append({
                    'interaction_id': f'I{len(interactions):08d}',
                    'user_id': user_id,
                    'offer_id': offer_id,
                    'type': 'click',
                    'timestamp': base_timestamp + timedelta(minutes=random.randint(1, 60))
                })
                
                # 30% chance of conversion after click
                if random.random() < 0.3:
                    interactions.append({
                        'interaction_id': f'I{len(interactions):08d}',
                        'user_id': user_id,
                        'offer_id': offer_id,
                        'type': 'conversion',
                        'timestamp': base_timestamp + timedelta(minutes=random.randint(61, 180))
                    })
        
        return pd.DataFrame(interactions).set_index('interaction_id')
    
    def save_to_csv(self, 
                    demographics: pd.DataFrame,
                    transactions: pd.DataFrame,
                    offers: pd.DataFrame,
                    interactions: pd.DataFrame,
                    output_dir: str = 'data/'):
        """Save generated data to CSV files."""
        demographics.to_csv(f'{output_dir}user_demographics.csv')
        transactions.to_csv(f'{output_dir}transactions.csv')
        offers.to_csv(f'{output_dir}offers.csv')
        interactions.to_csv(f'{output_dir}interactions.csv')
        
    def load_from_csv(self, input_dir: str = 'data/') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data from CSV files."""
        demographics = pd.read_csv(f'{input_dir}user_demographics.csv', index_col=0)
        transactions = pd.read_csv(f'{input_dir}transactions.csv', index_col=0)
        offers = pd.read_csv(f'{input_dir}offers.csv', index_col=0)
        interactions = pd.read_csv(f'{input_dir}interactions.csv', index_col=0)
        
        # Convert timestamp strings back to datetime
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
        
        return demographics, transactions, offers, interactions 