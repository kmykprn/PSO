import inspect
import numpy as np
import matplotlib.pyplot as plt
from objective_function import objective_function, sphere_objective_function



class PSO:
    """
    粒子群最適化 (PSO) を実装するクラス。

    Attributes:
        objective_function (function): 目的関数
        num_particles (int)          : 粒子の数
        max_iter (int)               : ループ回数
        position_range (float)       : 粒子の初期位置の範囲
        velocity_range (float)       : 粒子の初速度の範囲
        dimension (int)              : 入力の次元数
        weight (float)               : 慣性係数（w）
        c1 (float)                   : pbest の影響度
        c2 (float)                   : gbest の影響度
    """

    def __init__(self, objective_function, num_particles=100, max_iter=1000,
                 position_range=3, velocity_range=0.1, dimension=2, weight=0.1, c1=0.01, c2=0.01):

        self.objective_function = objective_function
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.position_range = position_range
        self.velocity_range = velocity_range
        self.dimension = dimension
        self.weight = weight
        self.c1 = c1
        self.c2 = c2

        # `objective_function` の引数の数をチェック
        num_params = len(inspect.signature(objective_function).parameters)
        if num_params != dimension:
            raise ValueError(f"エラー：目的関数の引数の数は {num_params} ですが、入力の次元数は{self.dimension}です")

    def initialize_particle(self):
        """
        粒子の初期位置・速度を設定し、初期のpbestとgbestを決定する。

        Returns:
            positions (np.ndarray)      : 各粒子の初期位置 (dimension, num_particles)
            velocities (np.ndarray)     : 各粒子の初速度 (dimension, num_particles)
            pbest_positions (np.ndarray): 各粒子の個体最良位置 (dimension, num_particles)
            pbest_ratings (np.ndarray)  : 各粒子の個体最良評価値 (num_particles,)
            gbest_position (np.ndarray) : 全体で最良の粒子位置 (dimension,)
            gbest_rating (float)        : 全体で最良の評価値
        """

        # 粒子の初期位置と速度をランダムに設定
        positions = np.random.uniform(-self.position_range, self.position_range, (self.dimension, self.num_particles))
        velocities = np.random.uniform(-self.velocity_range, self.velocity_range, (self.dimension, self.num_particles))

        # pbestの初期化
        pbest_positions = np.copy(positions)
        pbest_ratings = self.objective_function(*positions)

        # gbestの初期化
        idx = np.argmin(pbest_ratings)
        gbest_position = np.copy(pbest_positions[:, idx])
        gbest_rating = np.copy(pbest_ratings[idx])

        return positions, velocities, pbest_positions, pbest_ratings, gbest_position, gbest_rating


    def search(self):
        """
        PSO のメインループを実行し、最適な解を探索する。

        Returns:
            result (np.ndarray)        : 各イテレーションの gbest の履歴 (2, max_iter)
            gbest_position (np.ndarray): 最終的な最適解の位置 (2,)
            gbest_rating (float)       : 最終的な最適解の評価値
        """
        result = np.zeros((2,self.max_iter))
        for t in range(self.max_iter):

            # 粒子を初期化
            if t == 0:
                positions, velocities, pbest_positions, pbest_ratings, gbest_position, gbest_rating = self.initialize_particle()

            # self.dimension*self.num_particles のランダム数のリストを一括生成
            r1 = np.random.rand(self.dimension, self.num_particles)
            r2 = np.random.rand(self.dimension, self.num_particles)
            
            # 速度の更新（ベクトルで一括計算）
            velocities = (
                self.weight * velocities +
                self.c1 * r1 * (pbest_positions - positions) +
                self.c2 * r2 * (gbest_position.reshape(self.dimension, 1) - positions)
            )
            
            # 位置の更新（ベクトルで一括計算）
            positions += velocities

            # 各粒子の新しい評価値を一括計算
            new_ratings = self.objective_function(*positions)

            # pbestの更新（新しい評価値が良い場合のみ）
            update_filter = new_ratings < pbest_ratings
            if np.any(update_filter):
                pbest_positions[:, update_filter] = positions[:, update_filter]
                pbest_ratings[update_filter] = new_ratings[update_filter]

            # gbestの更新（新しい評価値が良い場合のみ）
            idx = np.argmin(pbest_ratings)
            if pbest_ratings[idx] < gbest_rating:
                gbest_position = np.copy(pbest_positions[:, idx])
                gbest_rating = pbest_ratings[idx]

             # 現在のイテレーションとgbestの評価値を保存
            result[:,t] = [t,gbest_rating]

        return result, gbest_position, gbest_rating


if __name__ == '__main__':
    # 粒子群最適化（PSO）の初期設定
    NUM_PARTICLES = 100
    MAX_ITER = 1000
    POSITION_RANGE = 3
    VELOCITY_RANGE = 0.1
    DIMENSION = 3
    WEIGHT = 0.1
    C1 = 0.01
    C2 = 0.01

    pso = PSO(objective_function=sphere_objective_function,
              num_particles=NUM_PARTICLES,
              max_iter=MAX_ITER,
              position_range=POSITION_RANGE,
              velocity_range=VELOCITY_RANGE,
              dimension=DIMENSION,
              weight=WEIGHT,
              c1=C1,
              c2=C2
    )
    result, gbest_position, gbest_rating = pso.search()

    # 最適解の出力
    print("Global Best Position:",  gbest_position)
    print("Global Best Value:", gbest_rating)

    # 次元が2のときのみプロット（3次元以上は可視化できない）
    if DIMENSION == 2:
        plt.plot(result[0,:],result[1,:])
        plt.xlabel("Iteration")
        plt.ylabel("Best Objective Value")
        plt.title("PSO Optimization Progress")
        plt.show()