import logging
import numpy as np
from typing import List, Callable
from ..config.config import ConfigManager

class LogObserver:
    """Observer interface for logging."""
    def update(self, message: str, level: str) -> None:
        pass

class FileLogObserver(LogObserver):
    """Observer that writes logs to a file."""
    def __init__(self, filename: str):
        self.logger = logging.getLogger('file_logger')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def update(self, message: str, level: str) -> None:
        level_map = {
            'INFO': self.logger.info,
            'WARNING': self.logger.warning,
            'ERROR': self.logger.error,
            'DEBUG': self.logger.debug
        }
        level_map.get(level, self.logger.info)(message)

class ConsoleLogObserver(LogObserver):
    """Observer that writes logs to console."""
    def __init__(self):
        self.logger = logging.getLogger('console_logger')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def update(self, message: str, level: str) -> None:
        level_map = {
            'INFO': self.logger.info,
            'WARNING': self.logger.warning,
            'ERROR': self.logger.error,
            'DEBUG': self.logger.debug
        }
        level_map.get(level, self.logger.info)(message)

class Logger:
    """Singleton logger class implementing the Observer pattern."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.observers: List[LogObserver] = []
        self.config = ConfigManager()
        
        # Add default observers
        self.add_observer(ConsoleLogObserver())
        self.add_observer(FileLogObserver('app.log'))
    
    def add_observer(self, observer: LogObserver) -> None:
        """Add a new observer."""
        self.observers.append(observer)
    
    def remove_observer(self, observer: LogObserver) -> None:
        """Remove an observer."""
        self.observers.remove(observer)
    
    def log(self, message: str, level: str = 'INFO') -> None:
        """Log a message to all observers."""
        for observer in self.observers:
            observer.update(message, level)

    def log_array_info(self, array: np.ndarray, title: str = ""):
        """Log information about a numpy array."""
        message = f"\n\n{title}:\n"
        message += f"Shape: {array.shape}\n"
        message += f"Type: {array.dtype}\n"
        message += f"Sample data:\n{array[:5]}\n"
        self.info(message)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.log(message, 'INFO')
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.log(message, 'WARNING')
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, 'ERROR')
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.log(message, 'DEBUG') 