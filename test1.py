# process_a.py
from utils import SharedMemoryManager
import numpy as np
import time
import pickle
from PIL import Image
import multiprocessing as mp
import os

# 配置参数
IMAGE_SIZE = (8192, 8192, 3)
RESULT_SIZE = (512, 512)

def client_send_image(data, gpu_id):
    """客户端发送单次请求到指定GPU"""
    try:
        # 连接服务端创建的资源
        shm_img = SharedMemoryManager(f"image_data_{gpu_id}")
        shm_result = SharedMemoryManager(f"result_data_{gpu_id}")

        # 发送数据
        print(f"[A-{gpu_id}] 发送图像数据...")
        shm_img.write_data(data)

        # 等待结果
        print(f"[A-{gpu_id}] 等待处理结果...")
        shm_result.wait_done()
        result = shm_result.read_data()
        
        return result

    except Exception as e:
        print(f"[A-{gpu_id}] 通信错误: {str(e)}")
        raise

def worker_process(gpu_id, image_path, request_count):
    """工作进程函数，向指定GPU发送请求"""
    print(f"[A-{gpu_id}] 启动工作进程，目标GPU: {gpu_id}")
    
    try:
        # 加载测试图像
        test_image = np.array(Image.open(image_path).convert('RGB'))
        
        for i in range(request_count):
            data = {
                'endpoint': 'location_2d',
                'image': test_image,
                'text': f"human-{i}"  # 添加序号区分不同请求
            }
            
            t_start = time.time()
            result = client_send_image(data, gpu_id)
            elapsed = time.time() - t_start
            
            print(f"[A-{gpu_id}] 请求{i+1}/{request_count} 处理时间: {elapsed:.2f}秒")
            print(f"[A-{gpu_id}] 收到结果: {str(result)[:100]}...")  # 只打印前100字符
            
            time.sleep(0.5)  # 请求间隔
            
    except KeyboardInterrupt:
        print(f"[A-{gpu_id}] 用户中断")
    except Exception as e:
        print(f"[A-{gpu_id}] 错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 获取可用的GPU数量
    num_gpus = 8  # 假设有4个GPU，可根据实际修改
    image_path = "./images/human.jpg"
    requests_per_gpu = 3  # 每个GPU发送的请求数量
    
    print(f"启动客户端，将向{num_gpus}个GPU各发送{requests_per_gpu}个请求")
    
    # 创建进程池
    processes = []
    
    try:
        # 为每个GPU创建一个工作进程
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, image_path, requests_per_gpu)
            )
            p.daemon = False
            p.start()
            processes.append(p)
            print(f"启动GPU {gpu_id}的客户端进程")
        
        # 等待所有进程完成
        for p in processes:
            p.join()
            
    except KeyboardInterrupt:
        print("\n主进程收到终止信号，终止所有工作进程")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=1)
    
    print("客户端主进程退出")