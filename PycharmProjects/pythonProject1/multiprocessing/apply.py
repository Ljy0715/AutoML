import time
from multiprocessing import Pool


def run(msg):
    print('msg:%s' %msg)
    # 程序随眠3秒,
    time.sleep(3)
    print('end')


if __name__ == "__main__":
    print("开始执行主程序")
    start_time=time.time()
    # 使用进程池创建子进程
    size=3
    pool=Pool(size)
    print("开始执行子进程")
    for i in range(size):
        pool.apply(run,(i,))
    print("主进程结束耗时%s"%(time.time()-start_time))
