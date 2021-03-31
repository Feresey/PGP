#include "sides.hpp"

dim3_type side_tag_to_dim3_type(side_tag tag)
{
    switch (tag) {
    default:
    case LEFT:
    case RIGHT:
        return DIM3_TYPE_X;
    case TOP:
    case BOTTOM:
        return DIM3_TYPE_Z;
    case FRONT:
    case BACK:
        return DIM3_TYPE_Y;
    }
}

dim3_type layer_tag_to_dim3_type(layer_tag tag)
{
    switch (tag) {
    default:
    case LEFT_RIGHT:
        return DIM3_TYPE_X;
    case VERTICAL:
        return DIM3_TYPE_Y;
    case FRONT_BACK:
        return DIM3_TYPE_Z;
    }
}

layer_tag dim3_type_to_layer_tag(dim3_type type)
{
    switch (type) {
    default:
    case DIM3_TYPE_X:
        return LEFT_RIGHT;
    case DIM3_TYPE_Y:
        return FRONT_BACK;
    case DIM3_TYPE_Z:
        return VERTICAL;
    }
}

layer_tag side_tag_to_layer_tag(side_tag tag)
{
    switch (tag) {
    default:
    case LEFT:
    case RIGHT:
        return LEFT_RIGHT;
    case TOP:
    case BOTTOM:
        return VERTICAL;
    case FRONT:
    case BACK:
        return FRONT_BACK;
    }
}

std::pair<int, int> other_sizes(const BlockGrid& grid, layer_tag tag)
{
    switch (tag) {
    default:
    case LEFT_RIGHT:
        return { grid.bsize.z, grid.bsize.y };
    case VERTICAL:
        return { grid.bsize.y, grid.bsize.x };
    case FRONT_BACK:
        return { grid.bsize.z, grid.bsize.x };
    }
}
