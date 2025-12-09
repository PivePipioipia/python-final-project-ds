from src.base_loader import BaseDataLoader  
import os
import time
import requests
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv
import yaml
import logging
from pathlib import Path

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TMDbDataLoader(BaseDataLoader):
    """
    DataLoader kế thừa BaseDataLoader để thu thập dữ liệu phim từ TMDb API.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__(name="tmdb_loader")  

        self.api_key = os.getenv("TMDB_API_KEY")
        self.base_url = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")

        if not self.api_key:
            raise ValueError("Không tìm thấy TMDB_API_KEY trong file .env")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Không tìm thấy config: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config["data_collection"]
        self.movies_data: List[Dict] = []

        logger.info("TMDbDataLoader đã được khởi tạo thành công")


    def fetch_data(self, *args, start_year=None, end_year=None, **kwargs):
        """
        Override của BaseDataLoader.fetch_data()
        Gọi hàm fetch_movies()
        """
        self.fetch_movies(start_year=start_year, end_year=end_year)

    def save_data(self, filepath: str) -> None:
        """
        Override: lưu CSV.
        """
        return self.save_to_csv(filepath)

    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """
        Override: load CSV.
        """
        return TMDbDataLoader.load_from_csv(filepath)

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        if params is None:
            params = {}

        params["api_key"] = self.api_key
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            time.sleep(0.25)
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                logger.error("API key không hợp lệ.")
            elif response.status_code == 429:
                logger.warning("Rate limit — đợi 10 giây...")
                time.sleep(10)
                return self._make_request(endpoint, params)
            else:
                logger.error(f"HTTP Error: {e}")
            raise

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def fetch_movies_by_year(self, year: int) -> List[Dict]:
        logger.info(f"Đang fetch phim năm {year}...")
        movies = []
        page = 1
        max_movies = self.data_config["max_movies_per_year"]

        while len(movies) < max_movies:
            params = {
                "language": self.data_config["language"],
                "page": page,
                "primary_release_year": year,
                "sort_by": "revenue.desc",
                "include_adult": "false",
                "with_original_language": self.data_config["language"],
            }

            try:
                data = self._make_request("/discover/movie", params)
                results = data.get("results", [])
                if not results:
                    break

                movies.extend(results)

                if page >= data.get("total_pages", 1):
                    break

                page += 1

            except Exception as e:
                logger.error(f"Lỗi trang {page} năm {year}: {e}")
                break

        return movies[:max_movies]

    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        try:
            return self._make_request(f"/movie/{movie_id}")
        except Exception as e:
            logger.error(f"Lỗi fetch chi tiết {movie_id}: {e}")
            return None

    def fetch_movies(self, start_year=None, end_year=None):
        if start_year is None:
            start_year = self.data_config["start_year"]
        if end_year is None:
            end_year = self.data_config["end_year"]

        logger.info(f"Fetch phim từ {start_year}–{end_year}")

        all_movies = []
        for year in range(start_year, end_year + 1):
            all_movies.extend(self.fetch_movies_by_year(year))

        detailed_movies = []
        min_budget = self.data_config["min_budget"]
        min_revenue = self.data_config["min_revenue"]

        for movie in tqdm(all_movies, desc="Fetching details"):
            details = self.get_movie_details(movie["id"])
            if not details:
                continue

            if details.get("budget", 0) >= min_budget and details.get("revenue", 0) >= min_revenue:
                detailed_movies.append(details)

        self.movies_data = detailed_movies

    def _extract_genres(self, genres_list):
        return "|".join([g["name"] for g in genres_list]) if genres_list else ""

    def _extract_companies(self, companies_list):
        return "|".join([c["name"] for c in companies_list]) if companies_list else ""

    def to_dataframe(self) -> pd.DataFrame:
        if not self.movies_data:
            raise ValueError("Chưa fetch dữ liệu!")

        processed = []
        for movie in self.movies_data:
            processed.append({
                "id": movie.get("id"),
                "title": movie.get("title"),
                "budget": movie.get("budget", 0),
                "revenue": movie.get("revenue", 0),
                "runtime": movie.get("runtime", 0),
                "release_date": movie.get("release_date"),
                "vote_average": movie.get("vote_average", 0),
                "vote_count": movie.get("vote_count", 0),
                "popularity": movie.get("popularity", 0),
                "overview": movie.get("overview", ""),
                "genres": self._extract_genres(movie.get("genres", [])),
                "production_companies": self._extract_companies(
                    movie.get("production_companies", [])
                ),
                "original_language": movie.get("original_language", ""),
                "status": movie.get("status", "")
            })

        return pd.DataFrame(processed)

    def save_to_csv(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(filepath, index=False, encoding="utf-8")
        logger.info(f"Đã lưu {len(df)} phim vào {filepath}")

    @staticmethod
    def load_from_csv(filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows từ {filepath}")
        return df

    def __repr__(self):
        return f"TMDbDataLoader(movies={len(self.movies_data)})"


if __name__ == "__main__":
    loader = TMDbDataLoader()
    loader.fetch_data(start_year=2020, end_year=2021)
    loader.save_data("data/raw/movies.csv")
