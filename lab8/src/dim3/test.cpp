#include <algorithm>

#include "dim3.hpp"

void test()
{
    mydim3<int> data = { 1, 2, 3 };
    auto max_elem = std::max_element(data.begin(), data.end());
    int max_dim = std::distance(data.begin(), max_elem);

    const auto& val = *max_elem;
}
