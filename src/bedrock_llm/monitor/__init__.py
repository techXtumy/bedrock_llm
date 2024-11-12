import time
import psutil
import pytz
import logging
from datetime import datetime
from termcolor import cprint
from functools import wraps


def _get_performance_metrics(func, start_time, start_memory):
    end_time = time.perf_counter()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    execution_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    return {
        "function": func.__name__,
        "start_time": start_time,
        "duration": execution_time,
        "memory_used": memory_used
    }

def _print_metrics(metrics):
    cprint("\n" + "="*50, "blue")
    cprint(f"[Performance Metrics]", "blue")
    cprint(f"Function    : {metrics['function']}", "blue")
    cprint(f"Start Time  : {metrics['start_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}", "blue")
    cprint(f"Duration    : {metrics['duration']:.2f} seconds", "blue")
    cprint(f"Memory Used : {metrics['memory_used']:.2f} MB", "blue")
    cprint("="*50, "blue")

def monitor_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_datetime = datetime.now(vietnam_tz)

        try:
            result = await func(*args, **kwargs)
            metrics = _get_performance_metrics(func, start_datetime, start_memory)
            _print_metrics(metrics)
            return result
        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            cprint(f"\n[ERROR] {func.__name__} failed after {execution_time:.2f} seconds", "red")
            cprint(f"Error: {str(e)}", "red")
            raise e

    return wrapper

def monitor_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_datetime = datetime.now(vietnam_tz)

        try:
            result = func(*args, **kwargs)
            metrics = _get_performance_metrics(func, start_datetime, start_memory)
            _print_metrics(metrics)
            return result
        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            cprint(f"\n[ERROR] {func.__name__} failed after {execution_time:.2f} seconds", "red")
            cprint(f"Error: {str(e)}", "red")
            raise e

    return wrapper

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def log_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__name__)
        logger.info(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = await func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully. Result: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise e
    return wrapper

def log_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__name__)
        logger.info(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully. Result: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise e
    return wrapper

# Setup logging when the module is imported
setup_logging()