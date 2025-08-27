# process_a.py
from utils import SharedMemoryManager
import numpy as np
import time
import pickle
from PIL import Image

# 配置参数
IMAGE_SIZE = (8192, 8192, 3)
RESULT_SIZE = (512, 512)

def client_send_image(data):
    """客户端发送单次请求"""
    try:
        # 连接服务端创建的资源
        shm_img = SharedMemoryManager("image_data")
        shm_result = SharedMemoryManager("result_data")

        # 发送数据
        print("[A] 发送图像数据...")
        shm_img.write_data(data)

        # 等待结果
        print("[A] 等待处理结果...")
        shm_result.wait_done()
        result = shm_result.read_data()
        
        return result

    except Exception as e:
        print(f"[A] 通信错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 模拟发送5次请求
    try:
        # 生成测试图像
        test_image = np.array(Image.open("./images/human.jpg").convert('RGB'))

        data = {
            'endpoint': 'location_2d',  # 指定服务端点
            'image': test_image,  # 转换为字节流
            'text': "human"
        }
        # 发送并获取结果
        t = time.time()
        result = client_send_image(data)
        print(f"[A] 请求处理时间: {time.time() - t:.2f}秒")
        print(f"[A] 收到结果")
        print(result)
        
        time.sleep(1)  # 请求间隔
        
    except KeyboardInterrupt:
        print("[A] 用户中断")
        exit()