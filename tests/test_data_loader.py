import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import os
import shutil
from src.base_loader import BaseDataLoader
from src.data_loader import TMDbDataLoader

class TestTMDbDataLoader(unittest.TestCase):

    def setUp(self):
        # Create a dummy config file if it doesn't exist (assuming project structure)
        # But here we heavily mock, so maybe not strictly needed if we mock __init__ properly
        # However, to be safe and integration-like, we let it read config if present.
        self.test_output_dir = "tests/test_data"
        os.makedirs(self.test_output_dir, exist_ok=True)
        self.loader = TMDbDataLoader()
        self.loader.data_config['max_movies_per_year'] = 2 # Limit for testing

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_inheritance(self):
        """Test if TMDbDataLoader inherits from BaseDataLoader"""
        print("\nTesting Inheritance...")
        is_subclass = issubclass(TMDbDataLoader, BaseDataLoader)
        self.assertTrue(is_subclass, "TMDbDataLoader should be a subclass of BaseDataLoader")
        is_instance = isinstance(self.loader, BaseDataLoader)
        self.assertTrue(is_instance, "Loader instance should be an instance of BaseDataLoader")
        print("Inheritance check passed.")

    @patch('src.data_loader.TMDbDataLoader._make_request')
    def test_fetch_and_save_data(self, mock_make_request):
        """Test fetch_data and save_data methods"""
        print("\nTesting Fetch and Save Data...")
        
        # Mock API responses
        # 1. Discover movies response
        mock_discover_response = {
            'results': [
                {'id': 1, 'title': 'Test Movie 1', 'release_date': '2020-01-01'},
                {'id': 2, 'title': 'Test Movie 2', 'release_date': '2020-01-02'}
            ],
            'total_pages': 1
        }
        
        # 2. Movie details response (called twice)
        mock_details_response_1 = {
            'id': 1, 'title': 'Test Movie 1', 'budget': 1000000, 'revenue': 2000000,
            'genres': [{'name': 'Action'}], 'production_companies': [{'name': 'Studio A'}]
        }
        mock_details_response_2 = {
            'id': 2, 'title': 'Test Movie 2', 'budget': 500000, 'revenue': 100000, # revenue < min_revenue likely
            'genres': [{'name': 'Drama'}], 'production_companies': [{'name': 'Studio B'}]
        }
        
        # Side effect to return different values based on call args or sequence
        # Call 1: discover, Call 2: detail 1, Call 3: detail 2
        mock_make_request.side_effect = [
            mock_discover_response,
            mock_details_response_1,
            mock_details_response_2
        ]

        # Override config filters for testing
        self.loader.data_config['min_budget'] = 0
        self.loader.data_config['min_revenue'] = 0

        self.loader.fetch_data(start_year=2020, end_year=2020)
        
        self.assertTrue(len(self.loader.movies_data) > 0, "Should satisfy fetching criteria")
        
        output_file = f"{self.test_output_dir}/test_movies.csv"
        self.loader.save_data(output_file)
        
        self.assertTrue(os.path.exists(output_file), "Output file should be created")
        
        # Test loading back
        df = TMDbDataLoader.load_data(output_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        print("Fetch, Save, and Load check passed.")

if __name__ == '__main__':
    unittest.main()
