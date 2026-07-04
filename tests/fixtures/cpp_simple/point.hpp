#pragma once

namespace geo {

class Point {
public:
    Point(int x, int y);
    Point();
    ~Point();

    int getX() const;
    int getY() const;

    void move(int dx, int dy);
    void move(double dx, double dy);

    Point operator+(const Point& other) const;

private:
    int x_;
    int y_;
};

}  // namespace geo
