import psutil
import time

def get_memory_usage():
    """Gets the system memory usage in MB."""

    mem = psutil.virtual_memory()
    return mem.used / (1024 ** 2)  # Convert bytes to MB

n = 0
while n<120:
    mem_usage = get_memory_usage()
    print(f"Memory usage: {mem_usage} MB")
    time.sleep(1)  # Check every 1 second
    n=n+1