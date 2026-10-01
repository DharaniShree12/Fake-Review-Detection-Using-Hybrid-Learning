"""Data Loader for AmazonDS Labeled Dataset"""
import pandas as pd
import numpy as np

class AmazonDSDataLoader:
    def __init__(self, data_path="AmazonDS/train.csv", num_reviews=2000):
        self.data_path = data_path
        self.num_reviews = num_reviews
        
    def load_data(self):
        """Load AmazonDS labeled review data from CSV"""
        print(f"Loading {self.num_reviews} reviews from {self.data_path}...")
        
        # Read CSV without header
        # Format: [label, title, text]
        # Labels: 1 = Fake, 2 = Real
        df = pd.read_csv(
            self.data_path,
            header=None,
            names=['label_raw', 'Title', 'Text'],
            nrows=self.num_reviews
        )
        
        print(f"Loaded {len(df)} reviews")
        print(f"Columns: {df.columns.tolist()}")
        
        # Standardize column names and labels
        df = self._standardize_columns(df)
        
        # Print statistics
        self._print_statistics(df)
        
        return df
    
    def _standardize_columns(self, df):
        """Standardize column names to match the system"""
        
        # Convert labels: 1 (Fake) → 1, 2 (Real) → 0
        df['label'] = df['label_raw'].apply(lambda x: 0 if x == 2 else 1)
        
        print(f"Original labels: {df['label_raw'].unique()}")
        print(f"Converted labels: {df['label'].unique()}")
        print(f"  1 (Fake) -> 1, 2 (Real) -> 0")
        
        # Generate synthetic user and product IDs
        # Create diverse users (each user writes ~2-3 reviews)
        df['UserId'] = 'User_' + (df.index // 2).astype(str)
        df['ProductId'] = 'Product_' + (df.index % 500).astype(str)
        
        # Add missing columns
        df['ProfileName'] = df['UserId']
        df['HelpfulnessNumerator'] = 0
        df['HelpfulnessDenominator'] = 0
        df['Score'] = np.random.choice([3, 4, 5], size=len(df))  # Random ratings
        df['Summary'] = df['Title']
        
        # Ensure proper types
        df['UserId'] = df['UserId'].astype(str)
        df['ProductId'] = df['ProductId'].astype(str)
        
        return df
    
    def _print_statistics(self, df):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("AMAZONDS LABELED DATASET STATISTICS")
        print("="*60)
        
        # Label distribution
        fake_count = sum(df['label'] == 1)
        real_count = sum(df['label'] == 0)
        print(f"Total Reviews: {len(df)}")
        print(f"  Real: {real_count} ({real_count/len(df)*100:.1f}%)")
        print(f"  Fake: {fake_count} ({fake_count/len(df)*100:.1f}%)")
        
        # Balance check
        balance_ratio = min(fake_count, real_count) / max(fake_count, real_count)
        if balance_ratio < 0.5:
            print(f"  WARNING: Imbalanced dataset (ratio: {balance_ratio:.2f})")
            print(f"  -> Using class weights (recommended)")
        else:
            print(f"  Well-balanced dataset (ratio: {balance_ratio:.2f})")
        
        # Text statistics
        if 'Text' in df.columns:
            avg_length = df['Text'].str.split().str.len().mean()
            min_length = df['Text'].str.split().str.len().min()
            max_length = df['Text'].str.split().str.len().max()
            print(f"\nText Statistics:")
            print(f"  Average Length: {avg_length:.1f} words")
            print(f"  Min Length: {min_length} words")
            print(f"  Max Length: {max_length} words")
        
        # User/Product statistics
        if 'UserId' in df.columns:
            unique_users = df['UserId'].nunique()
            print(f"\nGraph Statistics:")
            print(f"  Unique Users: {unique_users}")
            print(f"  Avg Reviews/User: {len(df)/unique_users:.2f}")
        
        if 'ProductId' in df.columns:
            unique_products = df['ProductId'].nunique()
            print(f"  Unique Products: {unique_products}")
            print(f"  Avg Reviews/Product: {len(df)/unique_products:.2f}")
        
        print("="*60 + "\n")
    
    def create_synthetic_labels(self, df):
        """Not needed - labels already exist"""
        print("Using REAL LABELS from AmazonDS dataset")
        print("This is a LABELED dataset - expect MUCH better accuracy!")
        return df


# Test function
def test_loader():
    """Test the AmazonDS data loader"""
    print("Testing AmazonDS Data Loader...")
    print("="*60)
    
    try:
        loader = AmazonDSDataLoader(num_reviews=100)
        df = loader.load_data()
        df = loader.create_synthetic_labels(df)
        
        print("\n✓ SUCCESS! Data loaded correctly.")
        print(f"  Loaded {len(df)} reviews")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Sample review: {df['Text'].iloc[0][:100]}...")
        print(f"  Sample label: {df['label'].iloc[0]}")
        
        return True
    
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_loader()
