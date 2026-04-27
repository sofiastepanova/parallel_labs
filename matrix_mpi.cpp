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
