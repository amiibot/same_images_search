import os
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk

# 设置tensorflow日志级别为ERROR，只显示错误信息
tf.get_logger().setLevel('ERROR')

# 加载预训练模型
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    try:
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))
        features = model.predict(img_array, verbose=0).flatten()
        return features
    except Exception:
        return None

def batch_process_images(image_files, folder_path, batch_size=100):
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i + batch_size]
        yield batch

def process_image(args):
    filepath, model = args
    return filepath, extract_features(filepath)

def find_model_duplicates(folder_path, output_dir="results", threshold=0.95, batch_size=100, num_threads=4):
    global model  # 使用全局模型实例
    file_paths = []
    features = []
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp']]
    total_images = len(image_files)
    print(f"找到 {total_images} 个图片文件")
    
    # 使用多线程提取图片特征
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for batch in batch_process_images(image_files, folder_path, batch_size):
            batch_paths = [os.path.join(folder_path, filename) for filename in batch]
            futures = [executor.submit(process_image, (path, model)) for path in batch_paths]
            
            for future in tqdm(futures, desc="处理图片批次", ncols=100, unit="张"):
                filepath, feat = future.result()
                if feat is not None:
                    features.append(feat)
                    file_paths.append(filepath)
    
    # 计算相似度矩阵
    similarity = cosine_similarity(features)
    
    # 分组重复图片
    duplicates = []
    visited = set()
    for i in range(len(similarity)):
        if i not in visited:
            group = [file_paths[i]]
            for j in range(i+1, len(similarity)):
                if similarity[i][j] > threshold:
                    group.append(file_paths[j])
                    visited.add(j)
            if len(group) > 1:
                duplicates.append(group)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建文本记录文件
    output_file = os.path.join(output_dir, "model_duplicates.txt")
    with open(output_file, 'w') as f:
        for idx, group in enumerate(duplicates, 1):
            # 为每组创建一个文件夹
            group_dir = os.path.join(output_dir, f"group_{idx}")
            os.makedirs(group_dir, exist_ok=True)
            
            # 复制图片到分组文件夹
            for img_path in group:
                filename = os.path.basename(img_path)
                dst_path = os.path.join(group_dir, filename)
                shutil.copy2(img_path, dst_path)
            
            # 写入文本记录
            f.write(f"相似图片组 {idx}:\n" + "\n".join(group) + "\n\n")
        
        # 计算统计信息
        total_groups = len(duplicates)
        total_images = sum(len(group) for group in duplicates)
        
        # 写入统计信息
        f.write("\n统计信息:\n")
        f.write(f"相似图片组总数: {total_groups}\n")
        f.write(f"所有组的图片总数: {total_images}\n")

class ImageSimilarityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图片相似度检测工具")
        
        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 参数设置区域
        ttk.Label(main_frame, text="相似度阈值 (0-1):").grid(row=0, column=0, sticky=tk.W)
        self.threshold_var = tk.StringVar(value="0.95")
        threshold_entry = ttk.Entry(main_frame, textvariable=self.threshold_var, width=10)
        threshold_entry.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(main_frame, text="批处理大小:").grid(row=1, column=0, sticky=tk.W)
        self.batch_size_var = tk.StringVar(value="100")
        batch_size_entry = ttk.Entry(main_frame, textvariable=self.batch_size_var, width=10)
        batch_size_entry.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(main_frame, text="线程数:").grid(row=2, column=0, sticky=tk.W)
        self.threads_var = tk.StringVar(value="4")
        threads_entry = ttk.Entry(main_frame, textvariable=self.threads_var, width=10)
        threads_entry.grid(row=2, column=1, sticky=tk.W)
        
        # 开始按钮
        start_button = ttk.Button(main_frame, text="开始处理", command=self.start_processing)
        start_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # 进度显示
        self.progress_var = tk.StringVar()
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        progress_label.grid(row=4, column=0, columnspan=2)
    
    def start_processing(self):
        try:
            threshold = float(self.threshold_var.get())
            batch_size = int(self.batch_size_var.get())
            num_threads = int(self.threads_var.get())
            
            if not (0 <= threshold <= 1):
                raise ValueError("阈值必须在0到1之间")
            
            self.progress_var.set("正在处理...")
            images_dir = os.path.join(os.path.dirname(__file__), "images")
            results_dir = os.path.join(os.path.dirname(__file__), "results")
            
            find_model_duplicates(images_dir, results_dir, threshold, batch_size, num_threads)
            self.progress_var.set("处理完成！结果保存在results目录中。")
            
        except ValueError as e:
            self.progress_var.set(f"错误：{str(e)}")
        except Exception as e:
            self.progress_var.set(f"发生错误：{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSimilarityGUI(root)
    root.mainloop()