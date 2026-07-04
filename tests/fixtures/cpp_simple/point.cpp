#include "point.hpp"

namespace geo {

Point::Point(int x, int y) : x_(x), y_(y) {}

Point::Point() : x_(0), y_(0) {}

Point::~Point() {}

int Point::getX() const {
    return x_;
}

int Point::getY() const {
    return y_;
}

void Point::move(int dx, int dy) {
    x_ += dx;
    y_ += dy;
}

void Point::move(double dx, double dy) {
    x_ += static_cast<int>(dx);
    y_ += static_cast<int>(dy);
}

Point Point::operator+(const Point& other) const {
    return Point(x_ + other.x_, y_ + other.y_);
}

}  // namespace geo
