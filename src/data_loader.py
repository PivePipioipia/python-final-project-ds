"""
Module TMDb Data Loader

Module chứa class TMDbDataLoader để thu thập dữ liệu phim từ TMDb API.
Class có khả năng tự động fetch dữ liệu theo năm, xử lý pagination,
lọc phim theo tiêu chí, và lưu vào file CSV.
"""

import os
import time
import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
import yaml
import logging
from pathlib import Path

# Tải các biến môi trường từ .env
# Giúp giấu đi API key đi, tránh viết thẳng vào code, an toàn hơn.
load_dotenv()

# Thiết lập hệ thống ghi log để theo dõi tiến trình và lỗi của chương trình
# Các log sẽ được in ra theo định dạng: [Thời gian] - [Tên file] - [Mức độ] - [Thông báo]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Tạo một logger riêng cho file này, giúp biết log đên từ đâu
logger = logging.getLogger(__name__)


class TMDbDataLoader:
    """
    Class để thu thập dữ liệu phim từ TMDb API.
    
    Class chứa các phương thức để:
    - Kết nối với TMDb API bằng API key
    - Fetch danh sách phim theo năm và các tiêu chí lọc
    - Lấy chi tiết của từng phim (budget, revenue, genres, overview...)
    - Lưu dữ liệu thô vào file CSV
    
    Attributes:
        api_key (str): API key của TMDb
        base_url (str): Base URL của TMDb API
        config (dict): Configuration từ file config.yaml
        movies_data (List[Dict]): Danh sách dữ liệu phim đã thu thập 
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Khởi tạo TMDbDataLoader với API key và configuration.
        
        Args:
            config_path (str): Đường dẫn đến file config.yaml
            
        Raises:
            ValueError: Nếu không tìm thấy API key trong environment variables
            FileNotFoundError: Nếu không tìm thấy file config
        """
        # Load API credentials từ .env
        self.api_key = os.getenv("TMDB_API_KEY")
        self.base_url = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
        
        if not self.api_key:
            raise ValueError(
                "Không tìm thấy TMDB_API_KEY trong file .env. "
                "Hãy tạo file .env từ .env.example và điền API key."
            )
        
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy file config: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data_collection']
        self.movies_data: List[Dict] = []
        
        logger.info("TMDbDataLoader đã được khởi tạo thành công")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Phương thức nội bộ này sẽ: Thực hiện HTTP request đến TMDb API và xử lý các lỗi phát sinh.
        
        Args:
            endpoint (str): API endpoint (ví dụ: "/discover/movie")
            params (Dict, optional): Query parameters cho request
            
        Returns:
            Dict: JSON response từ API
            
        Raises:
            requests.exceptions.RequestException: Nếu request thất bại
        """
        if params is None:
            params = {}
        
        # Thêm API key vào params
        params['api_key'] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Rate limiting: TMDb cho phép 40 requests/10 giây
            time.sleep(0.25)
            
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:  # Unauthorized
                logger.error("API key không hợp lệ. Vui lòng kiểm tra lại.")
            elif response.status_code == 429:  # Rate limit exceeded
                logger.warning("Vượt quá rate limit. Đợi 10 giây...")
                time.sleep(10)
                return self._make_request(endpoint, params)
            else:
                logger.error(f"HTTP Error: {e}") # các lỗi HTTP khác
            raise
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def fetch_movies_by_year(self, year: int) -> List[Dict]:
        """
        Fetch danh sách phim theo năm cụ thể với pagination.
        
        Args:
            year (int): Năm phát hành phim (ví dụ: 2020)
            
        Returns:
            List[Dict]: Danh sách các phim trong năm đó
        """
        logger.info(f"Đang fetch phim năm {year}...")
        
        movies = []
        page = 1
        max_movies = self.data_config['max_movies_per_year']
        
        while len(movies) < max_movies:
            params = {
                'language': self.data_config['language'],
                'page': page,
                'primary_release_year': year,
                'sort_by': 'revenue.desc',  # Sắp xếp theo revenue giảm dần
                'include_adult': 'false',
                'with_original_language': self.data_config['language']
            }
            
            try:
                data = self._make_request("/discover/movie", params)
                
                results = data.get('results', [])
                if not results:
                    break
                
                movies.extend(results)
                
                # Kiểm tra xem còn trang nữa không
                if page >= data.get('total_pages', 1):
                    break
                
                page += 1
                
            except Exception as e:
                logger.error(f"Lỗi khi fetch trang {page} của năm {year}: {e}")
                break
        
        logger.info(f"Đã fetch được {len(movies)} phim năm {year}")
        return movies[:max_movies]
    
    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """
        Lấy thông tin chi tiết của một phim cụ thể.
        
        Thông tin chi tiết bao gồm: budget, revenue, genres, overview,
        runtime, production_companies, và các thông tin khác.
        
        Args:
            movie_id (int): ID của phim trên TMDb
            
        Returns:
            Dict: Thông tin chi tiết của phim, hoặc None nếu lỗi
        """
        try:
            data = self._make_request(f"/movie/{movie_id}")
            return data
        except Exception as e:
            logger.error(f"Lỗi khi fetch chi tiết phim ID {movie_id}: {e}")
            return None
    
    def fetch_movies(
        self, 
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> None:
        """
        Fetch dữ liệu phim trong khoảng thời gian từ start_year đến end_year.
        
        Phương thức thực hiện:
        1. Fetch danh sách phim theo từng năm
        2. Lấy chi tiết từng phim (budget, revenue, genres...)
        3. Lọc phim theo tiêu chí min_budget và min_revenue
        4. Lưu vào attribute movies_data
        
        Args:
            start_year (int, optional): Năm bắt đầu. Mặc định từ config.
            end_year (int, optional): Năm kết thúc. Mặc định từ config.
        """
        if start_year is None:
            start_year = self.data_config['start_year']
        if end_year is None:
            end_year = self.data_config['end_year']
        
        logger.info(f"Bắt đầu fetch dữ liệu phim từ {start_year} đến {end_year}")
        
        all_movies = []
        
        # Fetch movies theo từng năm
        for year in range(start_year, end_year + 1):
            movies_in_year = self.fetch_movies_by_year(year)
            all_movies.extend(movies_in_year)
        
        logger.info(f"Tổng cộng fetch được {len(all_movies)} phim")
        logger.info("Đang lấy thông tin chi tiết cho từng phim...")
        
        # Lấy chi tiết từng phim với progress bar
        detailed_movies = []
        min_budget = self.data_config['min_budget']
        min_revenue = self.data_config['min_revenue']
        
        for movie in tqdm(all_movies, desc="Fetching movie details"):
            movie_id = movie['id']
            details = self.get_movie_details(movie_id)
            
            if details is None:
                continue
            
            # Lọc phim theo budget và revenue
            budget = details.get('budget', 0)
            revenue = details.get('revenue', 0)
            
            if budget >= min_budget and revenue >= min_revenue:
                detailed_movies.append(details)
        
        self.movies_data = detailed_movies
        logger.info(
            f"Sau khi lọc (budget >= {min_budget}, revenue >= {min_revenue}), "
            f"còn {len(detailed_movies)} phim hợp lệ"
        )
    
    def _extract_genres(self, genres_list: List[Dict]) -> str:
        """
        Chuyển đổi list of genres thành string phân cách bởi '|'.
        
        Args:
            genres_list (List[Dict]): List các genre dict từ API
            
        Returns:
            str: String genres phân cách bởi '|' (ví dụ: "Action|Adventure")
        """
        if not genres_list:
            return ""
        return "|".join([genre['name'] for genre in genres_list])
    
    def _extract_companies(self, companies_list: List[Dict]) -> str:
        """
        Chuyển đổi list of production companies thành string.
        
        Args:
            companies_list (List[Dict]): List các company dict từ API
            
        Returns:
            str: String companies phân cách bởi '|'
        """
        if not companies_list:
            return ""
        return "|".join([company['name'] for company in companies_list])
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Chuyển đổi movies_data thành pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu phim với các cột cần thiết
            
        Raises:
            ValueError: Nếu chưa có dữ liệu (cần gọi fetch_movies() trước)
        """
        if not self.movies_data:
            raise ValueError(
                "Chưa có dữ liệu phim. Vui lòng gọi fetch_movies() trước."
            )
        
        # Chọn và xử lý các cột cần thiết
        processed_data = []
        
        for movie in self.movies_data:
            processed_movie = {
                'id': movie.get('id'),
                'title': movie.get('title'),
                'budget': movie.get('budget', 0),
                'revenue': movie.get('revenue', 0),
                'runtime': movie.get('runtime', 0),
                'release_date': movie.get('release_date'),
                'vote_average': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0),
                'popularity': movie.get('popularity', 0),
                'overview': movie.get('overview', ''),
                'genres': self._extract_genres(movie.get('genres', [])),
                'production_companies': self._extract_companies(
                    movie.get('production_companies', [])
                ),
                'original_language': movie.get('original_language', ''),
                'status': movie.get('status', '')
            }
            processed_data.append(processed_movie)
        
        df = pd.DataFrame(processed_data)
        logger.info(f"Đã tạo DataFrame với shape: {df.shape}")
        
        return df
    
    def save_to_csv(self, filepath: str) -> None:
        """
        Lưu dữ liệu phim vào file CSV.
        
        Args:
            filepath (str): Đường dẫn file CSV để lưu
            
        Raises:
            ValueError: Nếu chưa có dữ liệu
        """
        # Tạo thư mục nếu chưa tồn tại
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        logger.info(f"Đã lưu {len(df)} phim vào file: {filepath}")
    
    @staticmethod
    def load_from_csv(filepath: str) -> pd.DataFrame:
        """
        Load dữ liệu phim từ file CSV.
        
        Args:
            filepath (str): Đường dẫn file CSV
            
        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu phim
            
        Raises:
            FileNotFoundError: Nếu file không tồn tại
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
        
        df = pd.read_csv(filepath, encoding='utf-8')
        logger.info(f"Đã load {len(df)} phim từ file: {filepath}")
        
        return df
    
    def __repr__(self) -> str:
        """String representation của TMDbDataLoader."""
        return (
            f"TMDbDataLoader("
            f"movies_count={len(self.movies_data)}, "
            f"year_range={self.data_config['start_year']}-{self.data_config['end_year']}"
            f")"
        )


if __name__ == "__main__":
    # Khởi tạo loader
    loader = TMDbDataLoader()
    
    # Fetch movies từ 2020-2024
    loader.fetch_movies(start_year=2020, end_year=2024)
    
    # Lưu vào CSV
    loader.save_to_csv("data/raw/movies_2020_2024.csv")
    
    # Load từ CSV đã có
    # df = TMDbDataLoader.load_from_csv("data/raw/movies_2020_2024.csv")
    # print(df.head())