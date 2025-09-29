"""
Utility functions for segmentation transformations
"""

from typing import Callable, Any
from functools import wraps
import time

from snowflake.snowpark import DataFrame
from loguru import logger


def log_transformation(func: Callable) -> Callable:
    """
    Decorator to log transformation execution details

    Args:
        func: The transformation function to wrap

    Returns:
        Wrapped function with logging
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Extract meaningful information for logging
        func_name = func.__name__

        # Get DataFrame from args (usually first argument after config)
        df_arg_position = 1 if len(args) > 1 and isinstance(args[1], DataFrame) else 0
        if df_arg_position < len(args) and isinstance(args[df_arg_position], DataFrame):
            input_df = args[df_arg_position]
            try:
                input_count = input_df.count()
                logger.info(f"Starting {func_name} | Input rows: {input_count:,}")
            except:
                logger.info(f"Starting {func_name} | Input DataFrame provided")
        else:
            logger.info(f"Starting {func_name}")

        # Execute the transformation
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Log output information
            if isinstance(result, DataFrame):
                try:
                    output_count = result.count()
                    logger.success(f"Completed {func_name} | Output rows: {output_count:,} | Execution time: {execution_time:.2f}s")
                except:
                    logger.success(f"Completed {func_name} | Execution time: {execution_time:.2f}s")
            else:
                logger.success(f"Completed {func_name} | Execution time: {execution_time:.2f}s")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func_name} | Error: {str(e)} | Execution time: {execution_time:.2f}s")
            raise

    return wrapper
