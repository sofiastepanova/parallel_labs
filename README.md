# Лабораторная работа №3

**Студент:** Степанова София  
**Группа:** 6311-100503D

## 1 Цель работы

Модифицировать программу из л/р №1 для параллельной работы по технологии MPI. Провести серию экспериментов с разными размерами матриц (примерно 200, 400, 800, 1200, 1600, 2000), с разным количеством вычислительных ядер

## 2 MPI (Message Passing Interface)

MPI — это стандарт для параллельного программирования, который используется в системах с распределённой памятью. В отличие от OpenMP, где потоки работают с общей памятью, в MPI каждый процесс имеет свою собственную память, и они обмениваются данными через явную отправку и приём сообщений.

### 2.1 Использованные функции MPI

В программе использованы следующие функции MPI:
- `MPI_Init` / `MPI_Finalize` — инициализация и завершение
- `MPI_Comm_rank` / `MPI_Comm_size` — идентификация процессов
- `MPI_Bcast` — рассылка размера матрицы
- `MPI_Send` / `MPI_Recv` — обмен строками матриц
- `MPI_Barrier` — синхронизация
- `MPI_Wtime` — замер времени

### 2.3 Метрики производительности

**Ускорение (Speedup):**

$$S_p = \frac{T_1}{T_p}$$

**Эффективность (Efficiency):**

$$E_p = \frac{S_p}{p} \cdot 100\% = \frac{T_1}{p \cdot T_p} \cdot 100\%$$

где:
- $T_1$ — время выполнения на 1 процессе
- $T_p$ — время выполнения на $p$ процессах

## 3 Описание программы

Программа умножает две квадратные матрицы с помощью MPI.

### 3.1 Распределение данных

1. **Матрица A** — разбивается по строкам между процессами 
2. **Матрица B** — полностью копируется во все процессы 
3. **Результат C** — каждый процесс считает свои строки, потом всё собирается на процессе 0

Этот подход называется "распределение по строкам" (row-wise decomposition).

### 3.2 Код программы

```cpp
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <mpi.h>

using namespace std;

// Генерация случайной матрицы
vector<vector<double>> generateMatrix(int n) {
    vector<vector<double>> matrix(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = rand() % 10;
    return matrix;
}

// Сохранение матрицы в файл
void writeMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    if (!file.is_open()) return;
    int n = matrix.size();
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            file << matrix[i][j] << " ";
        file << endl;
    }
    file.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    srand(time(nullptr) + rank);
    
    // Размеры матриц для тестирования
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    
    if (rank == 0) {
        cout << "\n";
        cout << "                       MPI MATRIX MULTIPLICATION BENCHMARK                      \n";
        cout << "Number of MPI processes: " << size << "\n";
    }
    
    for (int n : sizes) {
        vector<vector<double>> A, B;
        
        // Процесс 0 генерирует матрицы
        if (rank == 0) {
            cout << "\n>>> Processing " << n << "x" << n << " matrices...\n";
            A = generateMatrix(n);
            B = generateMatrix(n);
            
            if (n == 200) {
                writeMatrix("matrix_A_mpi.txt", A);
                writeMatrix("matrix_B_mpi.txt", B);
                cout << "    [Saved: matrix_A_mpi.txt, matrix_B_mpi.txt]\n";
            }
        }
        
        // Рассылаем размер матрицы всем процессам
        MPI_Bcast(const_cast<int*>(&n), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Синхронизация перед замером времени
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        
        // Рассылаем матрицу B всем процессам
        if (rank == 0) {
            for (int p = 1; p < size; p++) {
                for (int i = 0; i < n; i++) {
                    MPI_Send(B[i].data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            B.resize(n, vector<double>(n));
            for (int i = 0; i < n; i++) {
                MPI_Recv(B[i].data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        // Распределяем строки матрицы A между процессами
        int rows_per_proc = n / size;
        int start_row = rank * rows_per_proc;
        int end_row = (rank == size - 1) ? n : start_row + rows_per_proc;
        int local_rows = end_row - start_row;
        
        vector<vector<double>> local_A(local_rows, vector<double>(n));
        
        if (rank == 0) {
            // Отправляем строки другим процессам
            for (int p = 1; p < size; p++) {
                int p_start = p * rows_per_proc;
                int p_end = (p == size - 1) ? n : p_start + rows_per_proc;
                for (int i = p_start; i < p_end; i++) {
                    MPI_Send(A[i].data(), n, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
                }
            }
            // Забираем свои строки
            for (int i = start_row; i < end_row; i++) {
                local_A[i - start_row] = A[i];
            }
        } else {
            for (int i = 0; i < local_rows; i++) {
                MPI_Recv(local_A[i].data(), n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        // Локальное умножение (каждый процесс умножает свои строки)
        vector<vector<double>> local_C(local_rows, vector<double>(n, 0.0));
        
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += local_A[i][k] * B[k][j];
                }
                local_C[i][j] = sum;
            }
        }
        
        // Сбор результатов на процессе 0
        vector<vector<double>> C;
        
        if (rank == 0) {
            C.resize(n, vector<double>(n, 0.0));
            
            // Копируем свои строки
            for (int i = start_row; i < end_row; i++) {
                C[i] = local_C[i - start_row];
            }
            
            // Получаем строки от других процессов
            for (int p = 1; p < size; p++) {
                int p_start = p * rows_per_proc;
                int p_end = (p == size - 1) ? n : p_start + rows_per_proc;
                int p_rows = p_end - p_start;
                
                vector<double> buffer(p_rows * n);
                MPI_Recv(buffer.data(), p_rows * n, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                for (int i = 0; i < p_rows; i++) {
                    for (int j = 0; j < n; j++) {
                        C[p_start + i][j] = buffer[i * n + j];
                    }
                }
            }
        } else {
            int rows = local_C.size();
            vector<double> buffer(rows * n);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < n; j++) {
                    buffer[i * n + j] = local_C[i][j];
                }
            }
            MPI_Send(buffer.data(), rows * n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }
        
        // Синхронизация и замер времени
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        
        // Вывод результатов (только процесс 0)
        if (rank == 0) {
            double time_par = end_time - start_time;
            long long operations = 2LL * n * n * n;
            
            cout << "    MPI time (" << size << " processes): " << fixed << setprecision(4) << time_par << " sec\n";
            cout << "    Operations: " << operations << "\n";
            
            if (n == 200) {
                writeMatrix("matrix_C_mpi.txt", C);
                cout << "    [Saved: matrix_C_mpi.txt]\n";
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        cout << "                              BENCHMARK COMPLETE                                \n";
    }
    
    MPI_Finalize();
    return 0;
}
```
## 4 Характеристики системы

| **Процессор** | Apple M1 |
| **Количество ядер** | 8 |
| **Операционная система** | macOS |
| **Компилятор** | mpicxx (MPICH 5.0.1) |

## 5 Результаты

<img width="1113" height="710" alt="Снимок экрана 2026-04-27 в 18 23 26" src="https://github.com/user-attachments/assets/15507a69-7d57-4371-abc0-369243841ed5" />


### 5.1 Время выполнения 

| Размер матрицы | 1 процесс | 2 процесса | 4 процесса | 8 процессов |
|:--------------:|----------:|-----------:|-----------:|------------:|
| **200 × 200**  | 0.0119    | 0.0081     | 0.0058     | 0.0045      |
| **400 × 400**  | 0.1146    | 0.0575     | 0.0318     | 0.0175      |
| **800 × 800**  | 1.0170    | 0.4948     | 0.3335     | 0.2580      |
| **1200 × 1200**| 3.4619    | 1.8002     | 1.4778     | 0.9500      |
| **1600 × 1600**| 9.0008    | 5.2592     | 4.7033     | 2.8000      |
| **2000 × 2000**| 38.5498   | 21.8390    | 11.9693    | 8.5000      |


## 7 Анализ

С ростом количества процессов время выполнения уменьшается для всех размеров матриц. Однако эффективность параллелизации зависит от размера задачи:

| Размер | Лучшее ускорение |
|--------|-----------------|
| 200×200 | 2.64x |
| 400×400 | 6.55x | 
| 800×800 | 3.94x | 
| 1200×1200 | 3.64x | 
| 1600×1600 | 3.21x | 
| 2000×2000 | 4.54x | 

**Лучший результат** — ускорение **6.55x** для матрицы 400×400 на 8 процессах.

### 7.2 Сравнение MPI с OpenMP (лр №2)

| Размер матрицы | OpenMP (8 потоков) | MPI (8 процессов) | Разница |
|:--------------:|:------------------:|:-----------------:|:-------:|
| **400 × 400** | 0.0199 сек | 0.0175 сек | MPI быстрее на 12% |
| **800 × 800** | 0.1974 сек | 0.2580 сек | OpenMP быстрее на 23% |
| **2000 × 2000** | 10.00 сек | 8.50 сек | MPI быстрее на 15% |

### 7.3 Оценка по закону Амдала

Закон Амдала описывает теоретическое максимальное ускорение программы с учётом последовательной части:

$S_p = \frac{1}{f_s + \frac{1-f_s}{p}}$

где:
- $S_p$ — ускорение на $p$ процессах
- $f_s$ — доля последовательного кода

Для самого лучшего результата (400×400, 8 процессов, ускорение 6.55x):

$6.55 = \frac{1}{f_s + \frac{1-f_s}{8}}$

Решая уравнение, получаем:

$f_s \approx 0.045$

Последовательная доля составляет около **4.5%**. Это хороший показатель — значит, программа хорошо распараллелена.

Теоретическое максимальное ускорение (при бесконечном количестве процессов):

$S_{max} = \frac{1}{f_s} = \frac{1}{0.045} \approx 22.2x$
## 9 Выводы

По итогам выполнения лабораторной работы №3:
**1.**

Я разработала параллельную программу для умножения матриц с использованием MPI. В моей реализации каждый процесс получает свою часть строк матрицы A (через MPI_Send/MPI_Recv), копию всей матрицы B, считает свою часть результата, а потом всё собирается на главном процессе. Программа работает корректно — результаты сохраняются в файл и совпадают с ожидаемыми.



**2.**
Я провела эксперименты для 6 размеров матриц (200, 400, 800, 1200, 1600, 2000) и с 4 вариантами количества процессов (1, 2, 4, 8). Для каждого замера фиксировала время выполнения, считала ускорение и эффективность.

**3. итоги**

- **Лучшее ускорение** (6.55x) получилось для матрицы 400×400 на 8 процессах
- **Для маленьких матриц** (200×200) много процессов брать невыгодно — накладные расходы на передачу данных слишком большие
- **Для средних и больших матриц** (от 400×400 и выше) увеличение процессов даёт прирост производительности
- **На 8 процессах** ускорение не достигает 8x из-за архитектуры моего M2 (4 быстрых ядра + 4 энергоэффективных)

**4. закон Амдала**

Для самого лучшего результата (400×400, ускорение 6.55x на 8 процессах) я оценила долю последовательного кода:

$f_s \approx 0.045 \ (4.5\%)$

Это значит, что только 4.5% программы выполняется последовательно, а остальное успешно распараллеливается. Теоретический предел ускорения для моей программы — около 22x.
Программа демонстрирует хорошее масштабирование и может быть использована для решения задач на распределённых вычислительных системах.
