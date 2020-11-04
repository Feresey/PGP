#include <stdio.h>
#include <stdlib.h>

#include "serialize.h"


int main() {
    uint32_t* data;
    uint32_t w, h;
    int err;
    err = read_image(stdin, &data, &w, &h);
    if (err != 0) {
        printf("Error reading image: %s\n", err);
        exit(err);
    }
    err = write_image(stdout, data, w, h);
    if (err != 0) {
        printf("Error writing image: %d\n", err);
        exit(err);
    }
    free(data);
    return 0;
}