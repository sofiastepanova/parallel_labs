import matplotlib.pyplot as plt

sizes = [200, 400, 800, 1200, 1600, 2000]
threads = [1, 2, 4, 8]

times = {
    200: [0.0137, 0.0068, 0.0041, 0.0021],
    400: [0.1116, 0.0578, 0.0292, 0.0199],
    800: [0.9852, 0.4937, 0.2780, 0.1974],
    1200: [3.4570, 1.7727, 1.1793, 0.7950],
    1600: [8.6760, 4.4462, 2.9427, 1.9792],
    2000: [35.2596, 23.1353, 13.4803, 10.0002],
}

plt.figure(figsize=(10, 6))

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange']
for i, size in enumerate(sizes):
    plt.plot(threads, times[size], marker='o', label=f'{size}×{size}', color=colors[i])

plt.xlabel('Количество потоков', fontsize=12)
plt.ylabel('Время выполнения (сек)', fontsize=12)
plt.title('Зависимость времени умножения матриц от количества потоков', fontsize=14)
plt.legend()
plt.grid(True)

plt.savefig('time_vs_threads.png', dpi=150, bbox_inches='tight')
print("График сохранён как 'time_vs_threads.png'")

plt.show()
