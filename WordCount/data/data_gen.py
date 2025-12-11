import os
import random
import sys

WORD_POOL = [
    "Hadoop", "MapReduce", "BigData", "Java", "Python", "Spark", "Scala", "Kafka",
    "Streaming", "Processing", "Cluster", "Distributed", "Algorithm", "Parallel",
    "Computing", "Framework", "Memory", "Storage", "Network", "Performance",
    "Analysis", "Machine", "Learning", "Deep", "Neural", "Network", "Cloud",
    "Virtualization", "Container", "Docker", "Kubernetes", "Serverless"
]

def generate_file(filepath, target_size):
    """生成指定大小的文本文件"""
    print(f"生成文件: {os.path.basename(filepath)} (目标大小: {target_size/(1024*1024):.1f}MB)")
    
    written = 0
    with open(filepath, 'w', encoding='utf-8') as f:
        while written < target_size:
            # 生成一行
            words_in_line = random.randint(5, 15)
            line_words = []
            
            for _ in range(words_in_line):
                line_words.append(random.choice(WORD_POOL))
            
            line = ' '.join(line_words) + '\n'
            f.write(line)
            written += len(line.encode('utf-8'))
            
            # 显示进度
            if target_size > 10*1024*1024 and written % (5*1024*1024) == 0:
                percent = (written / target_size) * 100
                print(f"  进度: {percent:.1f}%")
    
    actual_size = os.path.getsize(filepath)
    print(f"  √ 完成! 实际大小: {actual_size/(1024*1024):.2f}MB")
    return actual_size

def main():
    output_dir = "./test_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成不同大小的文件
    file_sizes = [
        ("wordcount_1k.txt", 1024),           # 1KB
        ("wordcount_1m.txt", 1024*1024),      # 1MB
        ("wordcount_10m.txt", 10*1024*1024),  # 10MB
        ("wordcount_100m.txt", 100*1024*1024) # 100MB
    ]
    
    for filename, size in file_sizes:
        filepath = os.path.join(output_dir, filename)
        generate_file(filepath, size)
    
    print("\n所有测试数据生成完成！")
    print(f"文件保存在: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()