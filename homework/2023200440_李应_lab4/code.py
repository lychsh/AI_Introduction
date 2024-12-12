import random

def monte_carlo_pi(num_samples):
    """
    通过蒙特卡洛方法计算π的近似值
    参数:
        num_samples:生成随机点的个数
    """
    points = 0    # 生成的随机点在圆内的点的数目
    for _ in range(num_samples):
        x, y = random.random(), random.random()  # 在[0, 1]区间内随机生成一个点
        if x**2 + y**2 <= 1:  # 如果点在圆内
            points += 1

    return (points / num_samples) * 4  # 圆面积与正方形面积的比乘以4即为π的近似值


if __name__ == "__main__":

    num_samples = 10000000 # 生成点的样本数
    estimate_pi = monte_carlo_pi(num_samples)
    print(f"点的样本数为: {num_samples}, π的近似值是：{estimate_pi}")