#include <byteswap.h>
#include <stdint.h>
#include <stdio.h>

uint32_t scan_4()
{
    uint32_t temp;
    scanf("%x", &temp);
    return __bswap_32(temp);
}

void print_arr(FILE *out, const int *arr, const uint32_t size)
{
    for (uint32_t i = 0; i < size; ++i)
    {
        fprintf(out, "%08x", __bswap_32(arr[i]));
        if (i != size - 1)
        {
            fprintf(out, " ");
        }
    }
    fprintf(out, "\n");
}
