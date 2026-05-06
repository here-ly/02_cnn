import time
import torch
from functools import wraps


class Timer:
    """CUDA-aware 计时器。

    有 CUDA 时使用 torch.cuda.Event 精确测量 GPU 时间（不受异步执行影响），
    无 CUDA 时退化为 time.perf_counter 测量 CPU 墙钟时间。

    用法:
        with Timer("forward") as t:
            output = model(data)
        print(f"{t.name}: {t.elapsed:.3f}s")   # GPU kernel 时间

        with Timer("epoch", sync=True) as t:
            trainer.train_epoch()
        print(f"{t.name}: {t.elapsed:.3f}s")   # 墙钟时间（含 sync 等待）
    """

    def __init__(self, name="block", sync=False):
        self.name = name
        self.has_cuda = torch.cuda.is_available()
        self.sync = sync
        self.elapsed = None

    def __enter__(self):
        if self.has_cuda and not self.sync:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.has_cuda and not self.sync:
            self._end_event.record()
            self._end_event.synchronize()
            self.elapsed = self._start_event.elapsed_time(self._end_event) / 1000.0
        else:
            if self.has_cuda:
                torch.cuda.synchronize()
            self.elapsed = time.perf_counter() - self._start
        print(f"[TIMER] {self.name}: {self.elapsed:.3f}s")
        return False


def timeit(func=None, *, sync=False):
    """装饰器：@timeit 统计函数墙钟时间。

    sync=True 时在调用前后执行 torch.cuda.synchronize()，
    确保 GPU 异步操作完成后再计时结束，得到真实耗时。
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = f(*args, **kwargs)
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(f"[TIMER] {f.__name__}: {elapsed:.3f}s")
            return result
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
