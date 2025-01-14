import os
from ..data.synthetic_data_generator import SyntheticDataGenerator

def main():
    """Generate and save synthetic data for testing."""
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize data generator
    generator = SyntheticDataGenerator(
        n_users=1000,          # 1000 users
        n_offers_mop=100,      # 100 MOP offers
        n_offers_wildfire=100, # 100 Wildfire offers
        n_interactions=10000   # 10000 interactions
    )
    
    # Generate synthetic data
    print("Generating synthetic data...")
    demographics, transactions, offers, interactions = generator.generate_all_data()
    
    # Save data to CSV files
    print("\nSaving data to CSV files...")
    generator.save_to_csv(
        demographics=demographics,
        transactions=transactions,
        offers=offers,
        interactions=interactions,
        output_dir=data_dir + '/'
    )
    
    # Print summary statistics
    print("\nData Generation Summary:")
    print(f"Users: {len(demographics)}")
    print(f"Transactions: {len(transactions)}")
    print(f"Offers: {len(offers)}")
    print(f"Interactions: {len(interactions)}")
    
    print("\nSample of generated data:")
    print("\nUser Demographics:")
    print(demographics.head())
    print("\nTransactions:")
    print(transactions.head())
    print("\nOffers:")
    print(offers.head())
    print("\nInteractions:")
    print(interactions.head())

if __name__ == "__main__":
    main() 