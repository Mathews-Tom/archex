#include "list.hpp"

#include <cstdlib>

namespace geo {

List::List() : head_(nullptr), size_(0) {}

void List::push(const Point& value) {
    ListNode* node = static_cast<ListNode*>(std::malloc(sizeof(ListNode)));
    node->value = value;
    node->next = head_;
    head_ = node;
    ++size_;
}

ListNode* List::head() const {
    return head_;
}

std::size_t List::size() const {
    return size_;
}

}  // namespace geo
