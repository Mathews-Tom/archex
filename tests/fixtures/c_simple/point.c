#include "point.h"

Point point_make(int x, int y) {
    Point p;
    p.x = x;
    p.y = y;
    return p;
}

static int square(int n) {
    return n * n;
}

int point_distance_squared(const Point *a, const Point *b) {
    return square(a->x - b->x) + square(a->y - b->y);
}
