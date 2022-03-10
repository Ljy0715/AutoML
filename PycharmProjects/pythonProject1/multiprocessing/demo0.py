import time
import os
from multiprocessing import Pool


def countdown(n):
    while n > 0:
        n -= 1


if __name__ == "__main__":
    count = 2e7
    start = time.time()
    # n_processes = os.cpu_count()
    n_processes = 8  # 进程数
    pool = Pool(processes=n_processes)  # 进程池
    for i in range(n_processes):
        pool.apply_async(countdown, (count//n_processes,))  # 启动多进程
        # print(count//n_processes)
        print(i)
    pool.close()  # 使进程池不能添加新任务
    pool.join()  # 等待进程结束
    print(time.time() - start)
