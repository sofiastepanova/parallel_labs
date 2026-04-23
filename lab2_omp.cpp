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
    cout << "+-----------------------------------------------------\n";
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
