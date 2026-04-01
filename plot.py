import matplotlib.pyplot as plt


sizes = [10, 50, 100, 350, 500]
times = [0.000045, 0.00270263,0.0115698,0.479115, 1.24026 ]
plt.plot(sizes, times, marker='o')
plt.title("Зависимость времени от размера матрицы")
plt.xlabel("Размер матрицы (n)")
plt.ylabel("Время выполнения (сек)")
plt.grid()

plt.savefig("graph.png")
plt.show()
