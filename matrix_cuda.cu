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
