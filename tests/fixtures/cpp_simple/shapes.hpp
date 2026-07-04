#pragma once

namespace geo::shapes {

struct Size {
    int width;
    int height;
};

int area(int width, int height);
double area(double width, double height);

}  // namespace geo::shapes
