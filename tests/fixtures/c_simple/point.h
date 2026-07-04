#ifndef POINT_H
#define POINT_H

typedef struct Point {
    int x;
    int y;
} Point;

typedef struct {
    double width;
    double height;
} Size;

Point point_make(int x, int y);
int point_distance_squared(const Point *a, const Point *b);

#endif /* POINT_H */
