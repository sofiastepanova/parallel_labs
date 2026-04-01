
# Лабораторная работа №1

**Студент:** София Степанова  
**Группа:** 6311-100503D

---

## 1. Цель работы

Изучить работу с файловым вводом-выводом на языке C++, реализовать алгоритм умножения квадратных матриц и исследовать зависимость времени выполнения программы от размера входных данных.

---

## 2. Теоретические сведения

Произведение двух квадратных матриц определяется следующим образом:
Для элементов результирующей матрицы:


$$C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$


где:

- $$A$$ и $$B$$  — исходные матрицы  
- $$C$$ — результирующая матрица
- $$n$$ - размерность матриц

Количество арифметических операций при перемножении матриц:

$$ 2n^3$$

Алгоритмическая сложность:


$$O(n^3)$$

---

## 3. Описание алгоритма

Программа выполняет следующие шаги:

1. Генерация двух квадратных матриц заданного размера  
2. Сохранение матриц в файлы `matrix_A.txt` и `matrix_B.txt`  
3. Перемножение матриц с использованием трёх вложенных циклов  
4. Сохранение результата в файл `matrix_C.txt`  
5. Измерение времени выполнения  
6. Вычисление количества операций  

---

## 4. Исходный код программы

```cpp
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <cstdlib>

using namespace std;

vector<vector<double>> generateMatrix(int n) {
    vector<vector<double>> matrix(n, vector<double>(n));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = rand() % 10;

    return matrix;
}


void writeMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open output file: " + filename);
    }

    int n = matrix.size();
    file << n << endl;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i][j] << " ";
        }
        file << endl;
    }
}

vector<vector<double>> multiply(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    int n
) {
    vector<vector<double>> C(n, vector<double>(n, 0));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}


int main() {
    try {
        srand(time(0));

        int n = 10;


        auto A = generateMatrix(n);
        auto B = generateMatrix(n);


        writeMatrix("matrix_A.txt", A);
        writeMatrix("matrix_B.txt", B);


        auto start = chrono::high_resolution_clock::now();

        auto C = multiply(A, B, n);

        auto end = chrono::high_resolution_clock::now();
        double time = chrono::duration<double>(end - start).count();

        writeMatrix("matrix_C.txt", C);

        cout << "----------------------------------" << endl;
        cout << "Matrix size: " << n << "x" << n << endl;
        cout << "Execution time: " << time << " seconds" << endl;
        cout << "Operations: " << 2LL * n * n * n << endl;
        cout << "Files saved: matrix_A.txt, matrix_B.txt, matrix_C.txt" << endl;
        cout << "----------------------------------" << endl;

    } catch (const exception& e) {
        cout << "Error: " << e.what() << endl;
    }

    return 0;
}


```

---
  ## 5. Формат входных данных

Файлы `matrix_A.txt` и `matrix_B.txt` формируются автоматически программой и имеют следующий вид:
```txt
3
1 2 3
4 5 6
7 8 9
```


Первая строка содержит размер матрицы, далее идут её элементы.

---

## 6. Формат выходных данных

Файл `matrix_C.txt` содержит результат перемножения:

```txt
3
...
...
...
```

---

## 7. Верификация результатов

Для проверки корректности вычислений использовался язык Python и библиотека NumPy.

```python
import numpy as np


A = np.loadtxt("/Users/sofiastepanova/Desktop/лабы_парал-е/matrix_A.txt", skiprows=1)
B = np.loadtxt("/Users/sofiastepanova/Desktop/лабы_парал-е/matrix_B.txt", skiprows=1)
C = np.loadtxt("/Users/sofiastepanova/Desktop/лабы_парал-е/matrix_C", skiprows=1)

C_true = A @ B

print("C++ result:")
print(C)

print("\nPython result:")
print(C_true)

if np.allclose(C, C_true):
    print("\nOK: results are correct ")
else:
    print("\nERROR: results differ ")
```

Результат выполнения программы:

```
OK: results are correct
```
Это подтверждает правильность реализации алгоритма.

## 8. Исследование программы

Для анализа производительности были проведены эксперименты с различными размерами матриц.

| Размер матрицы | Количество операций | Время выполнения (с) |
|----------------|----------------------|-----------------------|
| 10             | 2000              | 0.000045        |
| 50            | 250000               | 0.00270263          |
| 100            | 2000000              | 0.0115698        |
| 350          | 85750000             | 0.479115       |
| 500            | 250000000               | 1.24026         |


## 9. Анализ результатов

Полученные данные показывают, что с увеличением размерности матриц время выполнения программы значительно возрастает.

При увеличении размера матрицы примерно в 2 раза наблюдается увеличение времени выполнения примерно в 8 раз, что согласуется с теоретической оценкой сложности алгоритма \(O(n^3)\).

![image.png](https://raw.githubusercontent.com/bucketio/img18/main/2026/04/01/1775023083984-512b4d5f-16ce-46c2-a805-781ae612c52b.png 'график.png')

```python
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
```



## 10. Вывод

В ходе выполнения лабораторной работы была разработана программа на языке C++ для перемножения квадратных матриц.

Реализована автоматическая генерация входных данных, вычисление результата и запись его в файл. Проведена проверка корректности вычислений с использованием Python и библиотеки NumPy.

Экспериментально установлено, что время выполнения алгоритма зависит от размера матрицы и растёт пропорционально \(n^3\), что соответствует теоретической сложности.
