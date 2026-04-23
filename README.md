
# $$\color{purple}{\text{Лабораторная работа №2}}$$


**Студент:** $$\color{pink}{\text{София Степанова}}$$
**Группа:** $$\color{#AAFBDC}{\text{6311-100503D}}$$

---

## $$\color{pink}{\text{1. Цель работы}}$$

 Модифицировать программу из л/р №1 для параллельной работы по технологии OpenMP.  Провести серию экспериментов с разным количеством потоков, разными размерами матриц, с разным количеством вычислительных ядер при наличии технической возможности, иначе использовать фиксированное существующее количество вычислительных ядер


---



## $$\color{#AAFBDC}{\text{2. OpenMP}}$$ 

OpenMP — это технология, предназначенная для организации параллельных вычислений в программах на C/C++. Она позволяет распределять вычисления между потоками с помощью специальных директив.

---

### 2.1 Используемые средства

В программе применяются:

- `#pragma omp parallel for` — параллелизация цикла  
- `collapse(2)` — объединение вложенных циклов  
- `omp_set_num_threads()` — установка числа потоков  
- `omp_get_wtime()` — измерение времени  

---

### 2.2 Метрики

**Ускорение:**

\[
S_p = \frac{T_1}{T_p}
\]

**Эффективность:**

\[
E_p = \frac{S_p}{p} \cdot 100\%
\]

---

## $$\color{pink}{\text{3. Описание программы}}$$ 

Программа выполняет умножение квадратных матриц и автоматически проводит серию тестов.

Как модифицировалась 1 лаба (ключевое):
- Добавлен заголовочный файл OpenMP
- Добавлена директива распараллеливания
- Добавлена установка количества потоков
- Изменён замер времени


---

### 3.1 Генерация матриц

Матрицы заполняются случайными числами:

```cpp
matrix[i][j] = rand() % 10;

```
### 3.2 Параллельный алгоритм
Используется OpenMP для распараллеливания:
```cpp
#pragma omp parallel for collapse(2) schedule(static)
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[i][k] * B[k][j];
        }
        C[i][j] = sum;
    }
}
```
### 3.3 Исходный код

```cpp
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>

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

vector<vector<double>> multiplySequential(const vector<vector<double>>& A,
                                           const vector<vector<double>>& B,
                                           int n) {
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

vector<vector<double>> multiplyParallel(const vector<vector<double>>& A,
                                         const vector<vector<double>>& B,
                                         int n,
                                         int num_threads) {
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

bool verifyResult(const vector<vector<double>>& C1,
                  const vector<vector<double>>& C2,
                  int n) {
    double eps = 1e-9;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(C1[i][j] - C2[i][j]) > eps) {
                return false;
            }
        }
    }
    return true;
}

void printTableHeader() {
    cout << "\n";
    cout << "|   Size   | Threads  | Time (sec) | Speedup  | Efficiency |\n";
}

void printTableRow(int size, int threads, double time, double speedup, double efficiency, bool correct) {
    cout << "| " << setw(8) << size << " | "
         << setw(8) << threads << " | "
         << setw(10) << fixed << setprecision(4) << time << " | "
         << setw(8) << setprecision(2) << speedup << " | "
         << setw(10) << setprecision(1) << efficiency << "% |";
    if (correct) cout << " ✓";
    cout << "\n";
}

void printTableFooter() {
    cout << "-----------------------------------------------------\n";
}

int main() {
    srand(time(nullptr));
    
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    vector<int> thread_counts = {1, 2, 4, 8};
    
    int max_threads = omp_get_max_threads();
    
    cout << "\n";
    cout << "                    OPENMP MATRIX MULTIPLICATION BENCHMARK                      \n";
    cout << "Max available threads: " << max_threads << "\n";
    cout << "Testing threads: 1, 2, 4, 8\n";
    
    for (int n : sizes) {
        cout << "\n>>> Processing " << n << "x" << n << " matrices...\n";
        
        vector<vector<double>> A = generateMatrix(n);
        vector<vector<double>> B = generateMatrix(n);
        
        if (n == 200) {
            writeMatrix("matrix_A.txt", A);
            writeMatrix("matrix_B.txt", B);
            cout << "    [Saved: matrix_A.txt, matrix_B.txt]\n";
        }
        
        cout << "    Running sequential multiplication...\n";
        auto start_seq = chrono::high_resolution_clock::now();
        vector<vector<double>> C_seq = multiplySequential(A, B, n);
        auto end_seq = chrono::high_resolution_clock::now();
        double time_seq = chrono::duration<double>(end_seq - start_seq).count();
        long long operations = 2LL * n * n * n;
        
        cout << "    Sequential time: " << fixed << setprecision(4) << time_seq << " sec\n";
        cout << "    Operations: " << operations << "\n";
        
        printTableHeader();
        
        for (int t : thread_counts) {
            int actual_threads = min(t, max_threads);
            
            cout << "    Running with " << actual_threads << " threads...\n";
            
            double start_par = omp_get_wtime();
            vector<vector<double>> C_par = multiplyParallel(A, B, n, actual_threads);
            double end_par = omp_get_wtime();
            double time_par = end_par - start_par;
            
            double speedup = time_seq / time_par;
            double efficiency = (speedup / actual_threads) * 100.0;
            bool correct = verifyResult(C_seq, C_par, n);
            
            printTableRow(n, actual_threads, time_par, speedup, efficiency, correct);
        }
        
        printTableFooter();
        
        if (n == 200) {
            vector<vector<double>> C_final = multiplyParallel(A, B, n, 4);
            writeMatrix("matrix_C_openmp.txt", C_final);
            cout << "    [Saved: matrix_C_openmp.txt]\n";
        }
    }
    
    cout << "                              BENCHMARK COMPLETE                                \n";
    
    return 0;
}

```
## $$\color{#AAFBDC}{\text{4. Характеристики системы }}$$ 

- **Процессор:** Apple M1 
- **Количество ядер:** 8 
- **Оперативная память:** 8 ГБ  
- **Операционная система:** macOS
- **Компилятор:** Clang + libomp

##  $$\color{pink}{\text{ 5. Результаты }}$$ 



| Размер матрицы | 1 поток | 2 потока | 4 потока | 8 потоков |
|:--------------:|--------:|---------:|---------:|----------:|
| **200 × 200**  | 0.0137  | 0.0068   | 0.0041   | 0.0021    |
| **400 × 400**  | 0.1116  | 0.0578   | 0.0292   | 0.0199    |
| **800 × 800**  | 0.9852  | 0.4937   | 0.2780   | 0.1974    |
| **1200 × 1200**| 3.4570  | 1.7727   | 1.1793   | 0.7950    |
| **1600 × 1600**| 8.6760  | 4.4462   | 2.9427   | 1.9792    |
| **2000 × 2000**| 35.2596 | 23.1353  | 13.4803  | 10.0002   |

## $$\color{#AAFBDC}{\text{6. График }}$$ 

![image.png](https://raw.githubusercontent.com/bucketio/img7/main/2026/04/23/1776961466578-6d954b40-bd00-421f-873e-c39223c1bb55.png 'грааааафик.png')


## $$\color{pink}{\text{ 7. Анализ }}$$ 

По полученным данным можно сделать следующие наблюдения.

С увеличением числа потоков время выполнения уменьшается для всех размеров матриц. Это подтверждает эффективность распараллеливания.

#### Малые размеры (200–400)

- наблюдается почти линейное ускорение;
- переход от 1 к 8 потокам даёт ускорение примерно в 6–6.5 раза;
- накладные расходы минимальны.

#### Средние размеры (800–1200)

- ускорение остаётся значительным;
- эффективность начинает снижаться;
- прирост от увеличения числа потоков становится менее выраженным.

#### Большие размеры (1600–2000)

- сохраняется стабильное ускорение;
- однако рост производительности замедляется;
- для 2000×2000 ускорение составляет около 3.5 раза (35.26 → 10.00 сек);
- это связано с ограничениями памяти и пропускной способности системы.

---

##  $$\color{#AAFBDC}{\text{8. Вывод }}$$ 

В ходе выполнения лабораторной работы была разработана параллельная версия программы умножения матриц с использованием технологии OpenMP.

- Реализовано параллельное умножение матриц с использованием директивы #pragma omp parallel for collapse(2)
- Проведено исследование масштабируемости для 6 размеров матриц (200, 400, 800, 1200, 1600, 2000) с количеством потоков 1, 2, 4, 8


Основные выводы:

- Параллельная реализация умножения матриц показала заметное снижение времени выполнения по сравнению с последовательной версией.
- Увеличение числа потоков ускоряет вычисления;
- Максимальный эффект достигается на малых и средних размерах;
- При больших размерах эффективность снижается;


Таким образом, использование OpenMP оправдано для ресурсоёмких задач, однако требует учета особенностей архитектуры системы.
