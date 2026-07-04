#pragma once

#include <cstddef>

#include "point.hpp"

namespace geo {

struct ListNode {
    Point value;
    ListNode* next;
};

class List {
public:
    List();

    void push(const Point& value);
    ListNode* head() const;
    std::size_t size() const;

private:
    ListNode* head_;
    std::size_t size_;
};

}  // namespace geo
