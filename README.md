# Лабораторная работа №5  
## Параллельное перемножение матриц на суперкомпьютере

**Студент:** Степанова София 
**Группа:** 6311-100503D  

---

## 1 Цель работы

Освоить методы параллельных вычислений с использованием технологии MPI, реализовать алгоритм умножения квадратных матриц на многопроцессорной вычислительной системе, провести тестирование производительности программы и оценить эффективность распараллеливания. Параллельную версию программы на MPI необходимо также запустить на суперкомпьютере «Сергей Королёв».


## 2 Теоретические сведения

Для двух квадратных матриц размерности N×N результат вычисляется по формуле:

$$
C_{ij} = \sum_{k=1}^{N} A_{ik} \cdot B_{kj}
$$

Общее количество арифметических операций определяется выражением:

$$
2N^3
$$

Вычислительная сложность алгоритма:

$$
O(N^3)
$$

Для ускорения вычислений применяется параллельный подход, при котором вычислительная нагрузка распределяется между несколькими MPI-процессами.



## 3 Алгоритм

В ходе выполнения лабораторной работы была реализована MPI-программа, использующая распределение строк первой матрицы между процессами.

Основные этапы работы алгоритма:

1. Главный процесс формирует или считывает исходные матрицы.
2. Размерность матриц передаётся всем вычислительным процессам.
3. Матрица A разбивается на части и распределяется между процессами.
4. Матрица B копируется всем участникам вычислений.
5. Каждый процесс выполняет вычисление своей части результирующей матрицы.
6. Частичные результаты собираются на основном процессе.
7. Итоговая матрица записывается в выходной файл.



### 3.1 Исходный код программы


```cpp
#include "mpi.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

int ProcNum, ProcRank;

void generateMatrix(vector<double>& matrix, int size)
{
    for (int i = 0; i < size * size; i++)
        matrix[i] = rand() % 10;
}

void saveMatrix(const char* filename, const vector<double>& matrix, int size)
{
    ofstream file(filename);
    file << size << endl;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            file << matrix[i * size + j] << " ";
        file << endl;
    }
    file.close();
}

void transposeMatrix(vector<double>& matrix, int size)
{
    for (int i = 0; i < size; i++)
        for (int j = i + 1; j < size; j++)
            swap(matrix[i * size + j], matrix[j * size + i]);
}

void multiplyMPI(vector<double>& A, vector<double>& B, vector<double>& C, int size)
{
    int baseRows = size / ProcNum;
    int extraRows = size % ProcNum;

    vector<int> sendcounts(ProcNum), displs(ProcNum);
    int offset = 0;

    for (int i = 0; i < ProcNum; i++)
    {
        int rows = (i < extraRows) ? baseRows + 1 : baseRows;
        sendcounts[i] = rows * size;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int localRows = sendcounts[ProcRank] / size;

    vector<double> localA(sendcounts[ProcRank]);
    vector<double> localC(sendcounts[ProcRank]);

    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 localA.data(), sendcounts[ProcRank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Bcast(B.data(), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    transposeMatrix(B, size);

    for (int i = 0; i < localRows; i++)
        for (int j = 0; j < size; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < size; k++)
                sum += localA[i * size + k] * B[j * size + k];
            localC[i * size + j] = sum;
        }

    MPI_Gatherv(localC.data(), sendcounts[ProcRank], MPI_DOUBLE,
                C.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    srand(time(0) + ProcRank);

    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};

    if (ProcRank == 0)
    {
        cout << "MPI MATRIX MULTIPLICATION BENCHMARK" << endl;
        cout << "Number of MPI processes: " << ProcNum << endl;
    }

    for (int size : sizes)
    {
        vector<double> A, B, C;

        if (ProcRank == 0)
        {
            A.resize(size * size);
            B.resize(size * size);
            C.resize(size * size);

            generateMatrix(A, size);
            generateMatrix(B, size);
        }
        else
        {
            B.resize(size * size);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double start = MPI_Wtime();

        multiplyMPI(A, B, C, size);

        MPI_Barrier(MPI_COMM_WORLD);

        double end = MPI_Wtime();

        if (ProcRank == 0)
        {
            cout << "Processing " << size << "x" << size << endl;
            cout << "MPI time (" << ProcNum << " processes): "
                 << end - start << " sec" << endl;

            long long operations = 2LL * size * size * size;
            cout << "Operations: " << operations << endl;

            if (size == 200)
            {
                saveMatrix("matrix_A_mpi.txt", A, size);
                saveMatrix("matrix_B_mpi.txt", B, size);
                saveMatrix("matrix_C_mpi.txt", C, size);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
```
## 4. Результат выполнения

### Таблица времени выполнения для 8 процессов

| Размер матрицы | Количество процессов | Время выполнения (сек) |
|----------------|----------------------|------------------------|
| 200×200        | 8                    | 0.0038                 |
| 400×400        | 8                    | 0.0178                 |
| 800×800        | 8                    | 0.3641                 |
| 1200×1200      | 8                    | 0.9909                 |
| 1600×1600      | 8                    | 2.7058                 |
| 2000×2000      | 8                    | 5.6115                 |



## 5. Вывод

В рамках лабораторной работы была разработана и протестирована программа параллельного умножения матриц с использованием технологии MPI. 

В процессе выполнения работы были:

- изучены механизмы взаимодействия процессов в MPI;
- реализован алгоритм распределения вычислений;
- проведены экспериментальные измерения времени выполнения;
- подтверждена эффективность параллельного подхода.

Полученные результаты показывают, что применение нескольких вычислительных процессов позволяет существенно сократить время решения ресурсоёмких задач.
Таким образом, цель работы достигнута: освоены базовые принципы MPI и реализован эффективный параллельный алгоритм
