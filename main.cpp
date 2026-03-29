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
