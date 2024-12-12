import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


url = "http://bl.mmd.ac.cn:8889/image_query"
labels = ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#00FFFF", "#FF00FF", "#800000", "#008000", "#000080", "#808000",
         "#800080", "#008080", "#444444", "#FFD700", "#008080"]

def load_images_deocde(num):
    """
    获得dog和cat图片编码数据
    num: 图片数量
    """
    data = []
    for i in range(1, num + 1):
        time.sleep(0.5)     # 避免抓取过快
        try:
            with open(f"archive\\DogsCats\\Cats\\cat{i}.jpeg", "rb") as file:
                image = {"image": file}
                response = requests.post(url, files=image)   # 通过文澜api获得图像编码
                # print(response.json())
                data.append(np.array(response.json()["embedding"]))    # 保存图像编码到列表中
        except FileNotFoundError:
            print(f"文件archive\\DogsCats\\Cats\\cat{i}.jpeg不存在")
    for i in range(1, num + 1):
        time.sleep(0.5)
        try:
            with open(f"archive\\DogsCats\\Dogs\\dog{i}.jpeg", "rb") as file:
                image = {"image": file}
                response = requests.post(url, files=image)     # 通过文澜api获得图像编码
                # print(response.json())
                data.append(np.array(response.json()["embedding"]))     # 保存图像编码到列表中
        except FileNotFoundError:
            print(f"文件archive\\DogsCats\\Dogs\\dog{i}.jpeg不存在")
    return np.array(data)    # 返回ndarray二维数组

def save_data(data):
    """
    保存图像编码数据为npy格式
    """
    np.save("data.npy", data)

def load_data(path: str):
    """
    加载图像编码数据
    """
    data = np.load(path)
    return data


def init_random_data(n, sigma, mu):
    data = sigma * np.random.randn(n, 2) + mu
    return data

def init_centroids(data, k):
    """
    随机选择k个质心返回
    """
    indexs = np.random.choice(data.shape[0], k, replace=False)
    return data[indexs]

def init_centroids_plusplus(data, k):
    """
    kmeans++随机初始质心
    """
    centroids = [data[np.random.randint(data.shape[0])]]
    for _ in range(1, k):
        distances = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(data[j])
                break
    return np.array(centroids)

def cal_distance(centroids, point):
    """
    计算点point与各个质心之间的欧式距离，返回最近质心下标(point标签)
    """
    dis = []
    for centroid in centroids:
        dis.append(np.sqrt(np.sum((point - centroid) ** 2)))
    return np.argmin(dis)    # 返回最近的簇的下标


def cal_consine_similiraty(centroids, new_centroids):
    """
    计算质心前后的余弦相似度
    """
    cos_sims = []
    for i, centroid in enumerate(centroids):
        new_centroid = new_centroids[i]
        dot_product = np.dot(centroid, new_centroid)    # 点积
        cos_sim = dot_product / (np.linalg.norm(centroid) + np.linalg.norm(new_centroid))   # 计算余弦相似度
        cos_sims.append(cos_sim)
    return min(cos_sims)    # 返回最小的余弦相似度


def update_centroids(clusters):
    """
    计算每一簇的平均值更新质心
    """
    new_centroids = []
    for cluster in clusters:
        cluster = np.array(cluster)
        new_centroids.append(cluster.mean(axis=0))
    return np.array(new_centroids)

def Kmeans(centroids, k, maxdepth=10000, epsilon=0.98):
    """
    maxdepth: 最大迭代深度
    epsilon : 前后两次质心向量余弦相似度的阈值
    """
    depth = 0
    clusters = [[] for i in range(k)]    # k个簇
    while depth < maxdepth:
        # 分配点到每个簇
        for point in data:
            lable = cal_distance(centroids, point)
            clusters[lable].append(point)
        # 重新计算质心
        new_centroids = update_centroids(clusters)
        # 检查收敛
        if cal_consine_similiraty(centroids, new_centroids) > epsilon:   # 如果质心基本不在变化，结束迭代
            break
        # 更新质心
        centroids = new_centroids
        depth += 1
    return clusters, centroids


def visiual_clusters(clusters, k):
    """
        降维并可视化三维数组中的类别，每个类别用不同的颜色表示。

        参数:
        clusters -- 三维数组列表，每个二维数组是一个类别。
        color_labels -- 包含颜色代码的列表，用于为每个类别指定颜色。
        """
    plt.figure(figsize=(10, 8))
    # 遍历每个类别(二维数组)
    for index, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        # 绘制当前类别的数据点
        plt.scatter(cluster[:, 0], cluster[:, 1], s=50, c=labels[index], label=f'Class {index}')

    plt.legend()
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.show()



if __name__ == "__main__":
    k = 4
    sigma = 5
    mu = 6
    num = 300
    # file_num = 54
    # data = np.array(load_images_deocde(num))
    # save_data(data)
    # 使用DELL数据集
    data = load_data("data.npy")
    perplexity = min(k, len(data) - 1)
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    data = tsne.fit_transform(data)
    # 使用随机数据集
    # data = init_random_data(num, sigma, mu)
    centroids = init_centroids_plusplus(data, k)
    clusters, centroids = Kmeans(centroids, k)
    visiual_clusters(clusters, k)
    print("聚类质心为:", centroids)




