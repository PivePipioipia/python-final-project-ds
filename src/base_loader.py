from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

class BaseDataLoader(ABC):
    """
    Lớp trừu tượng cho (Abstract Base Class) định nghĩa giao diện chung cho mọi DataLoaer trong pipeline.

    Mục đích: 
    - Đảm bảo các loader có cùng cấu trúc phương thức, áp dụng kế thừa và đa hình
    - Giups pipeline xử lý dữ liệu 1 cách thống nhất và dễ dàng tái sử dụng

    Các phương thức bắt buộc:
    - fetch_data(): Thu thập dữ liệu thô
    - to_dataframe(): Chuyển dữ liệu thô thành DataFrame (pandas)
    - save_data(): Lưu dữ liệu vào file
    - load_data(): Load dữ liệu từ file

    Bất kỳ class con nào kế thừa BaseDataLoader đều phải override các phương thức này
    """

    def __init__(self, name: str = "base_loader") -> None:
        self.name = name

    @abstractmethod
    def fetch_data(self, *arg: Any, **kwargs: Any) -> None: 
        """
        Thu thập dữ liệu thô từ nguồn lưu vào thuộc tính nội bộ
        trả về None
        """
        raise NotImplementedError
    
    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """
        Chuyển dữ liệu thô thành DataFrame (pandas)
        trả về DataFrame
        """
        raise NotImplementedError

    @abstractmethod
    def save_data(self, filepath: str) -> None:
        """
        Lưu dữ liệu vào file
        trả về None
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_data(filepath:str) -> pd.DataFrame:
        """
        Load dữ liệu từ file
        trả về DataFrame
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


