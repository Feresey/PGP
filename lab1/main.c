#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void inverse(float* vec, int n)
{
    for (int i = 0; i < n; ++i) {
        vec[n - i] = vec[i];
    }
}

int main()
{
    int n;
    scanf("%d", &n);
    float* vec = (float*)(malloc(sizeof(float) * n));

    for (int i = 0; i < n; i++) {
        scanf("%f", &vec[i]);
    }

    time_t begin = clock();
    inverse(vec, n);
    time_t end = clock();

    // milliseconds
    fprintf(stderr, "cpu time = %f\n", (float)(end - begin) / (CLOCKS_PER_SEC / 1000));

    for (int i = 0; i < n; i++) {
        printf("%.10e ", vec[i]);
    }
    printf("\n");
}