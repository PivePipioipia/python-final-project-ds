import logging
import sys
from src.data_loader import TMDbDataLoader

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

def test_loader():
    print("Testing TMDbDataLoader with parallel execution...")
    try:
        loader = TMDbDataLoader()
        # Fetch a very small range
        loader.fetch_data(start_year=2023, end_year=2023)
        
        # Check if data was fetched
        if not loader.movies_data:
            print("❌ No data fetched!")
            return
            
        print(f"✅ Successfully fetched {len(loader.movies_data)} movies.")
        
        # Test saving
        output_file = "data/raw/test_movies.csv"
        loader.save_data(output_file)
        print(f"✅ Saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loader()
