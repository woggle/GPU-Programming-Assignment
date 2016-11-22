#include <assert.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <cfloat>
#include <utility>

// #define USE_EXAMPLE /* uncomment to use fixed example instead of generating vectors */
// #define PRINT_ANSWERS /* uncomment to print out solution */

const int threads_per_block = 256;


// Forward function declarations
/* Solve AX=B, where A is upper-triangular NxN, X and B are NxM. Returns the answer in X. B is not modified.
 *
 * Elements of X, B are stored in row-major order, so if B is 3x3, the elements are:
 * B[0] B[1] B[2]
 * B[3] B[4] B[5]
 * B[6] B[7] B[8]
 * i.e., if rows and columns are numbered starting with 0, then row i, column j is at index i*M + j
 *
 * Elements of A are stored packed:
 * So, if A is 5x5, then the elements of A are:
 * A[0]
 * A[1]  A[2]
 * A[3]  A[4]  A[5]
 * A[6]  A[7]  A[8]  A[9]
 * A[10] A[11] A[12] A[13]
 * A[14] A[15] A[16] A[17] A[18]
 *
 * In particular, if rows and columns are numbered starting with 0, then
 * row i, column j is at index (i)*(i+1)/2 + j.
 */
void GPU_forward_subst(float *A, float *X, float *B, int N, int M, int kernel_code, float *kernel_time, float *transfer_time);
void CPU_forward_subst(float *A, float *X, float *B, int N, int M);
float *get_random_vector(int N);
float *get_increasing_vector(int N);
float usToSec(long long time);
long long start_timer();
long long stop_timer(long long start_time, const char *name);
void printMatrix(float *X, int N, int M);
void die(const char *message);
void checkError();

// Main program
int main(int argc, char **argv) {

    //default kernel
    int kernel_code = 1;
    
    // Parse vector length and kernel options
    int N, M;
#ifdef USE_EXAMPLE
    if (argc == 3 && !strcmp(argv[1], "-k")) {
        kernel_code = atoi(argv[2]); 
        printf("KERNEL_CODE %d\n", kernel_code);
    } else
#endif
    if (argc == 3) {
        N = atoi(argv[1]); // user-specified value
        M = atoi(argv[2]); // user-specified value
    } else if (argc == 4 && !strcmp(argv[2], "-k")) {
        N = atoi(argv[1]); // user-specified value
        M = atoi(argv[2]); // user-specified value
        kernel_code = atoi(argv[3]); 
        printf("KERNEL_CODE %d\n", kernel_code);
    } else {
        die("USAGE: ./forward_subst <N> <M> -k <kernel_code> # AX=B, A is NxN, B is N rows x M cols");
    }

    // Seed the random generator (use a constant here for repeatable results)
    srand(10);

    // Generate random matrices
    long long vector_start_time = start_timer();
#ifdef USE_EXAMPLE
    /* for debugging, code to use a fixed example; uncomment above to enable */
    float A[6] = {
        2.0,
        3.0, 4.0,
        5.0, 6.0, 7.0
    };
    float B[6] = {
        2.0, 3.0,
        4.0, 5.0,
        6.0, 7.0,
    };
    N = 3;
    M = 2;
#else
    float *A = get_random_vector(N * N);
    /* to make it more likely to be numerical stable, have diagonal of A be larger,
       and grow in proportion to the row
     */
    for (int i = 0; i < N; ++i) {
        A[i * (i + 1) / 2 + i] *= (1 + i);
    }
    float *X_real = get_random_vector(N * M);
    float *B;
    cudaMallocHost((void **) &B, N * M * sizeof(float));
    memset(B, 0, N * M * sizeof(float));
    /* generate B from AX_real, so the answers should be in the range [1,2] */
    for (int i = 0; i < N; ++i) {
        float* A_row = &A[i * (i + 1) / 2];
        for (int k = 0; k < M; ++k) {
            for (int j = 0; j <= i; ++j) {
                B[i * M + k] += A_row[j] * X_real[j * M + k];
            }
        }
    }
#endif
    float *X_GPU;
    float *X_CPU;
    cudaMallocHost((void **) &X_GPU, N * M * sizeof(float));
    cudaMallocHost((void **) &X_CPU, N * M * sizeof(float));
    memset(X_CPU, 0, N * M * sizeof(float));
    memset(X_GPU, 0, N * M * sizeof(float));
    //float *vec = get_increasing_vector(N);
    stop_timer(vector_start_time, "Vector generation");
	
    // Compute the max on the GPU
    float GPU_kernel_time = INFINITY;
    float transfer_time = INFINITY;
    long long GPU_start_time = start_timer();
    GPU_forward_subst(A, X_GPU, B, N, M, kernel_code, &GPU_kernel_time, &transfer_time);
    long long GPU_time = stop_timer(GPU_start_time, "\t            Total");
	
    printf("%f\n", GPU_kernel_time);
    
    // Compute the max on the CPU
    long long CPU_start_time = start_timer();
    CPU_forward_subst(A, X_CPU, B, N, M);
    long long CPU_time = stop_timer(CPU_start_time, "\nCPU");
    
    // Free matrices 
    cudaFree(A);
    cudaFree(B);

    // Compute the speedup or slowdown
    //// Not including data transfer
    if (GPU_kernel_time > usToSec(CPU_time)) printf("\nCPU outperformed GPU kernel by %.2fx\n", (float) (GPU_kernel_time) / usToSec(CPU_time));
    else                     printf("\nGPU kernel outperformed CPU by %.2fx\n", (float) usToSec(CPU_time) / (float) GPU_kernel_time);

    //// Including data transfer
    if (GPU_time > CPU_time) printf("\nCPU outperformed GPU total runtime (including data transfer) by %.2fx\n", (float) GPU_time / (float) CPU_time);
    else                     printf("\nGPU total runtime (including data transfer) outperformed CPU by %.2fx\n", (float) CPU_time / (float) GPU_time);

#ifdef PRINT_ANSWERS
    printf("CPU result:\n");
    printMatrix(X_CPU, N, M);
    printf("GPU result:\n");
    printMatrix(X_GPU, N, M);
#endif

    // Check the correctness of the GPU results
    float max_delta = 0.0f;
    for (int i = 0; i < N * M; ++i) {
        float cpu = X_CPU[i];
        float gpu = X_GPU[i];
#ifndef USE_EXAMPLE
        float real = X_real[i];
        float delta = fabs(gpu - real) / real;
        assert(fabs(cpu - real) / real < 1e-6 * 2 * N);
#else
        float delta = fabs(gpu - cpu);
#endif
        if (delta > max_delta) {
            /* printf("%f/%f/%f\n", gpu, cpu, real); */
            max_delta = delta;
        }
    }
    cudaFree(X_CPU);
    cudaFree(X_GPU);
    /* This should be lenient enough to allow additions/substractions to occur in a different order */
    int wrong = max_delta > 1e-6 * 2 * N;
    // Report the correctness results
    if(wrong) printf("GPU output did not match CPU output (max error %.2f%%)\n", max_delta * 100.);
}

void GPU_forward_subst(float *A, float *X, float *B, int N, int M, int kernel_code, float *kernel_runtime, float *transfer_runtime) {
    // IMPLEMENT YOUR BFS AND TIMING CODE HERE
}


void CPU_forward_subst(float *A, float *X, float *B, int N, int M) {	
    cudaMemcpy(X, B, N * M * sizeof(float), cudaMemcpyHostToHost);
    for (int row = 0; row < N; ++row) {
        /* for debugging
        printMatrix(X, N, M);
        printf("-- (beginning of row %d) --\n", i);
        */
        float *A_row = &A[row * (row + 1) / 2];
        for (int j = 0; j < row; ++j) {
            for (int k = 0; k < M; ++k) {
                X[M * row + k] -= A_row[j] * X[M * j + k];
            }
        }
        /* for debugging 
        printMatrix(X, N, M);
        printf("--\n");
        */
        for (int k = 0; k < M; ++k) {
            X[M * row + k] /= A_row[row];
        }
    }
}


// Returns a randomized vector containing N elements
// This verison generates vector containing values in the range [1,2)
float *get_random_vector(int N) {
    if (N < 1) die("Number of elements must be greater than zero");
	
    // Allocate memory for the vector
    float *V;
    cudaMallocHost((void **) &V, N * sizeof(float));
    if (V == NULL) die("Error allocating CPU memory");
	
    // Populate the vector with random numbers
    for (int i = 0; i < N; i++) V[i] = 1.0f + rand() * 1.0f / RAND_MAX;
	
    // Return the randomized vector
    return V;
}

void printMatrix(float *X, int N, int M) {
    for (int i = 0; i < N; ++i) {
        printf("row %d: ", i);
        for (int j = 0; j < M; ++j) {
            printf("%f ", X[i * M + j]);
        }
        printf("\n");
    }
}

void checkError() {
    // Check for kernel errors
    cudaError_t error = cudaGetLastError();
    if (error) {
        char message[256];
        sprintf(message, "CUDA error: %s", cudaGetErrorString(error));
        die(message);
    }
}

// Returns the current time in microseconds
long long start_timer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// converts a long long ns value to float seconds
float usToSec(long long time) {
    return ((float)time)/(1000000);
}

// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, const char *name) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
    float elapsed = usToSec(end_time - start_time);
    printf("%s: %.5f sec\n", name, elapsed);
    return end_time - start_time;
}


// Prints the specified message and quits
void die(const char *message) {
    printf("%s\n", message);
    exit(1);
}
