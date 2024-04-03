#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define ROWS 70000
#define COLS 784

#include "mnist_tm_sparse.h"

double get_time_diff(struct timespec *start, struct timespec *end) {
    double start_sec = (double)start->tv_sec + ((double)start->tv_nsec / 1e9);
    double end_sec = (double)end->tv_sec + ((double)end->tv_nsec / 1e9);
    return end_sec - start_sec;
}


int main() {
   

     

    // Read x values
    FILE *x_file = fopen("mnist_x_70000_784.test_bin", "rb");
    if (x_file == NULL) {
        printf("Failed to open x file\n");
        return 1;
    }

    uint8_t *x_data = malloc(ROWS * COLS * sizeof(*x_data));
    if (x_data == NULL) {
        printf("Failed to allocate memory for x_data\n");
        return 1;
    }

    size_t x_read = fread(x_data, sizeof(uint8_t), ROWS * COLS, x_file);
    if (x_read != ROWS * COLS) {
        printf("Failed to read all data from x file\n");
        fclose(x_file);
        free(x_data);
        return 1;
    }

    fclose(x_file);

    // Read y values
    FILE *y_file = fopen("mnist_y_70000_784.test_bin", "rb");
    if (y_file == NULL) {
        printf("Failed to open y file\n");
        free(x_data);
        return 1;
    }

    uint32_t y_data[ROWS];

    size_t y_read = fread(y_data, sizeof(uint32_t), ROWS, y_file);
    if (y_read != ROWS) {
        printf("Failed to read all data from y file\n");
        fclose(y_file);
        free(x_data);
        return 1;
    }
    fclose(y_file);


    int correct = 0;
    int total = 0;
    struct timespec start, end;


    // for(int k = 0;k < ROWS; ++k)
    const int n_total_predictions = 1000000;
    int n_total_correct = 0;
    double total_time = 0.0;

    for(int k = 0; k < n_total_predictions; ++k)
    {
        int i = k % ROWS;
        uint8_t* example = &x_data[i*COLS];

        clock_gettime(CLOCK_MONOTONIC, &start);
        int y_hat = predict_tm(example);
        clock_gettime(CLOCK_MONOTONIC, &end);

        total_time += get_time_diff(&start, &end);
    
        if(y_hat == y_data[i])
            correct += 1;
        
        total += 1;

    }
 
    printf("correct: %d, total: %d \n", correct, total);
    printf("total time: %f\n", total_time);
    
    /*
    =====================================================================
    gcc -o tm.out mnist_test.c    
    -
    correct: 973990, total: 1000000 
    total time: 67.706492

    =====================================================================
    gcc -O3 -march=native -mtune=native -funroll-loops -fwhole-program -o tm.out mnist_test.c

    correct: 973990, total: 1000000 
    total time: 43.749774

    =====================================================================
    gcc -O3 -march=native -mtune=native -ftree-vectorize -ftree-vectorizer-verbose=2 -fopt-info-vec-optimized -fopt-info-vec-missed -o tm.out mnist_test.c

    correct: 973990, total: 1000000 
    total time: 54.481346

    =====================================================================
    gcc -O3 -march=native -mtune=native -o tm.out mnist_test.c

    correct: 973990, total: 1000000 
    total time: 54.149469

    =====================================================================
    gcc -O3 -mavx2 -mfma -ftree-vectorize -o tm.out mnist_test.c

    correct: 973990, total: 1000000 
    total time: 54.214224
    
    =====================================================================
    gcc -O3 -mavx2 -mfma -ftree-vectorize -o tm.out mnist_test.c
    [TOPOLOGICAL v1]

    correct: 973990, total: 1000000 
    total time: 26.547865

    =====================================================================
    gcc -O3 -march=native -mtune=native -funroll-loops -fwhole-program -o tm.out mnist_test.c
    [TOPOLOGICAL v1]

    total time: 27.188897

    =====================================================================
    gcc -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -fwhole-program -o tm.out mnist_test.c
    [TOPOLOGICAL v1]

    total time: 28.748274

    =====================================================================
    gcc -O3 -mavx2 -mfma -o tm.out mnist_test.c
    [TOPOLOGICAL v1]
    
    total time: 26.327967


    
    */    

    free(x_data);

    return 0;
}
