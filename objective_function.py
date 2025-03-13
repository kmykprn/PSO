import numpy as np
import matplotlib.pyplot as plt

# 目的関数
def objective_function(X,Y):
    t1 = 20
    t2 = -20 * np.exp(-0.2 * np.sqrt(1.0 / 2 * (X**2 + Y**2 )))
    t3 = np.e
    t4 = -np.exp(1.0 / 2 * (np.cos(2 * np.pi * X)+np.cos(2 * np.pi * Y)))
    return t1 + t2 + t3 + t4

# # Figureと3DAxeS
# fig = plt.figure(figsize = (8, 8))
# ax = fig.add_subplot(111, projection="3d")

# # 軸ラベルを設定
# ax.set_xlabel("x", size = 16)
# ax.set_ylabel("y", size = 16)
# ax.set_zlabel("z", size = 16)

# # 円周率の定義
# pi = np.pi

# # (x,y)データを作成
# x = np.linspace(-2*pi, 2*pi, 256)
# y = np.linspace(-2*pi, 2*pi, 256)

# # 格子点を作成
# X_p, Y_p = np.meshgrid(x, y)
# Z = objective_function(X_p,Y_p)

# # # 曲面を描画
# ax.plot_surface(X_p, Y_p, Z, cmap = "summer")

# # # 底面に等高線を描画
# ax.contour(X_p, Y_p, Z, colors = "black", offset = -1)

# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
