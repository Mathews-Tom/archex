#include <cstdio>

#include "list.hpp"
#include "platform.hpp"
#include "point.hpp"
#include "shapes.hpp"

int main() {
    geo::Point origin;
    geo::List list;
    list.push(origin);
    std::printf("%s\n", platform_name());
    return 0;
}
