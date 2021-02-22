#include <byteswap.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

uint32_t scan_4()
{
    uint32_t temp;
    // вот запустит потом какой-нибудь умник этот чудесный код на одноплатнике с другим порядком байт...
    assert(fread(&temp, sizeof(uint32_t), 1, stdin) == 1);
    return temp;
}

void print_arr(FILE *out, const int *arr, const uint32_t size)
{
    uint32_t i; // error: ‘for’ loop initial declarations are only allowed in C99 mode
    for (i = 0; i < size; ++i)
    {
        fprintf(out, "%08x", __bswap_32((uint32_t)arr[i]));
        if (i != size - 1)
        {
            fprintf(out, " ");
        }
    }
    fprintf(out, "\n");
}
