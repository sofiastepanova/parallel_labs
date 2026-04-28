# Лабораторная работа №4

**Студентка:** Степанова София  
**Группа:** 6311-100503D

## 1 Цель работы
 Модифицировать программу из л/р №1 для параллельной работы по технологии CUDA. Провести эксперименты с разными размерами матриц и различными конфигурациями сетки блоков.

## 2 CUDA

CUDA — это технология от NVIDIA, которая позволяет использовать видеокарту для вычислений. На GPU тысячи маленьких ядер, которые отлично справляются с однотипными операциями (например, умножение матриц). CPU делает это медленнее, потому что у него ядер мало (4-8), хотя каждое ядро мощное.

### 2.1 Как работает программа на CUDA

Программа запускает **Kernel** — специальную функцию, которая выполняется на GPU. Потоки организованы в блоки, а блоки — в сетку (grid).

| Компонент | значение |
|-----------|---------|
| **Thread** | Один поток, считает один элемент матрицы |
| **Block** | Группа потоков (например, 16×16 = 256 потоков) |
| **Grid** | Сетка блоков (зависит от размера матрицы) |

### 2.2 Метрики

**Ускорение:** 
$S_p = \frac{T_{CPU}}{T_{GPU}}$

**Эффективность:** 
$E = \frac{S_p}{p} \cdot 100\%$ (но для GPU это не совсем корректно, потому что у GPU тысячи ядер)

## 3 Программа

1. Генерирует случайные матрицы A и B (на CPU)
2. Копирует их в память GPU (VRAM)
3. Запускает CUDA ядро, где каждый поток считает один элемент результата
4. Копирует результат обратно в память CPU
5. Выводит время выполнения

### код
```
%%writefile matrix_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Ядро CUDA с shared memory
__global__ void matmul_tiled_kernel(float *A, float *B, float *C, int n) {
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * 32 + ty;
    int col = bx * 32 + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (n + 31) / 32; tile++) {
        if (row < n && tile * 32 + tx < n)
            tileA[ty][tx] = A[row * n + tile * 32 + tx];
        else
            tileA[ty][tx] = 0.0f;

        if (col < n && tile * 32 + ty < n)
            tileB[ty][tx] = B[(tile * 32 + ty) * n + col];
        else
            tileB[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < 32; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

void init_matrix(float *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = (float)(rand() % 10);
    }
}

void matmul_cpu(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    int sizes[] = {200, 400, 800, 1200, 1600, 2000};
    
    printf("\n");
    printf("================================================================\n");
    printf("               CUDA MATRIX MULTIPLICATION BENCHMARK             \n");
    printf("================================================================\n");
    
    int block_configs[][2] = {
        {16, 16},
        {32, 32},
        {8, 8}
    };
    
    for (int cfg = 0; cfg < 3; cfg++) {
        int bx = block_configs[cfg][0];
        int by = block_configs[cfg][1];
        printf("\n>>> Block: %d x %d (%d threads)\n", bx, by, bx*by);
        
        for (int s = 0; s < 6; s++) {
            int n = sizes[s];
            size_t bytes = n * n * sizeof(float);
            
            float *h_A = (float*)malloc(bytes);
            float *h_B = (float*)malloc(bytes);
            float *h_C_gpu = (float*)malloc(bytes);
            float *h_C_cpu = (float*)malloc(bytes);
            
            srand(42);
            init_matrix(h_A, n);
            init_matrix(h_B, n);
            
            float *d_A, *d_B, *d_C;
            cudaMalloc(&d_A, bytes);
            cudaMalloc(&d_B, bytes);
            cudaMalloc(&d_C, bytes);
            
            cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
            
            dim3 blockSize(bx, by);
            dim3 gridSize((n + bx - 1) / bx, (n + by - 1) / by);
            
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start);
            matmul_tiled_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
            cudaEventRecord(stop);
            cudaDeviceSynchronize();
            
            float time_ms = 0;
            cudaEventElapsedTime(&time_ms, start, stop);
            float time_gpu = time_ms / 1000.0f;
            
            cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
            
            cudaEventRecord(start);
            matmul_cpu(h_A, h_B, h_C_cpu, n);
            cudaEventRecord(stop);
            cudaDeviceSynchronize();
            
            float cpu_time_ms = 0;
            cudaEventElapsedTime(&cpu_time_ms, start, stop);
            float time_cpu = cpu_time_ms / 1000.0f;
            
            float speedup = time_cpu / time_gpu;
            
            printf("    %4d | GPU: %.4f sec | CPU: %.4f sec | Speedup: %.2fx\n",
                   n, time_gpu, time_cpu, speedup);
            
            free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            cudaEventDestroy(start); cudaEventDestroy(stop);
        }
    }
    
    printf("\n================================================================\n");
    printf("                      BENCHMARK COMPLETE                         \n");
    printf("================================================================\n");
    
    return 0;
}
```
### 3.1 Конфигураци
Три разных размера блоков:
- **8×8** (64 потока на блок)
- **16×16** (256 потоков на блок)
- **32×32** (1024 потока на блок)

Для каждого размера матрицы (200, 400, 800, 1200, 1600, 2000) программа запускалась несколько раз, и я записывала время.
### 3.2 характеристики

| Параметр | Значение |
|----------|----------|
| **GPU** | NVIDIA Tesla T4 (Google Colab) |
| **VRAM** | 16 GB |
| **CUDA версия** | 13.0 |

## 4 Результаты экспериментов

## графики
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/38cedcfe-c347-492e-abcb-ef4e0736cc70" />


### 4.1 Время выполнения на GPU (секунды)

| Размер | Блок 8×8 | Блок 16×16 | Блок 32×32 |
|:------:|---------:|-----------:|-----------:|
| 200 | 0.0001 | 0.0950 | 0.0001 |
| 400 | 0.0004 | 0.0002 | 0.0004 |
| 800 | 0.0034 | 0.0016 | 0.0025 |
| 1200 | 0.0114 | 0.0082 | 0.0090 |
| 1600 | 0.0269 | 0.0193 | 0.0201 |
| 2000 | 0.0526 | 0.0374 | 0.0401 |

### 4.2 Сравнение с CPU

| Размер | CPU время | GPU время | Ускорение |
|:------:|----------:|-------------------:|----------:|
| 200 | 0.0241 | 0.0001 | **322x** |
| 400 | 0.2105 | 0.0002 | **950x** |
| 800 | 1.9519 | 0.0016 | **1205x** |
| 1200 | 8.0864 | 0.0082 | **991x** |
| 1600 | 25.0545 | 0.0193 | **1301x** |
| 2000 | 64.1821 | 0.0374 | **1716x** |

**Лучшее ускорение: 1716 раз** (для 2000×2000, блок 16×16)

### 4.3 Влияние размера блока

| Размер | Лучший блок | Время | Почему |
|--------|-------------|-------|--------|
| 200 | 8×8 или 32×32 | 0.0001 сек | Разница не заметна из-за малого объёма |
| 400-800 | 16×16 | 0.0002-0.0016 сек | Хороший баланс |
| 1200-2000 | 16×16 | 0.0082-0.0374 сек | Оптимальная загрузка ядер |

## 5 Анализ


**Для маленькой матрицы (200×200):**
- GPU не показал супер-ускорения
- Накладные расходы на копирование данных в память GPU больше, чем выгода

**Для средних и больших матриц (400+):**
- Ускорение безумное — от 950 до 1716 раз
- GPU справляется за тысячные доли секунды, пока CPU думает

**Про блоки:**
- **16×16** оказался самым удачным для больших матриц
- **8×8** и **32×32** работают тоже хорошо, но чуть медленнее

### 5.1 Закон Амдала

Для лучшего результата (2000×2000, ускорение 1716x):

$S_{max} = \frac{1}{f_s} \approx 1716$  
$f_s \approx 0.00058$ (0.06%)

Последовательная доля программы — всего 0.06%. Это значит, что почти весь код распараллелился.

## 6 Выводы

По итогам лабораторной работы №4:

Разработана параллельная версия программы умножения матриц с использованием CUDA. Реализовано ядро, где каждый поток вычисляет один элемент результирующей матрицы.

Проведено исследование для матриц размером от 200 до 2000 с конфигурациями блоков 8×8, 16×16 и 32×32.

ускорение до 1716 раз для матрицы 2000×2000

Программа демонстрирует, что использование GPU с CUDA даёт колоссальное ускорение по сравнению с CPU и является эффективным решением для задач линейной алгебры.
