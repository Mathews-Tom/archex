#pragma once

namespace geo {

template <typename T>
class Container {
public:
    void add(T item) {
        items_[count_++] = item;
    }

    void add(const T& item, int count) {
        for (int i = 0; i < count; ++i) {
            items_[count_++] = item;
        }
    }

    int size() const {
        return count_;
    }

private:
    T items_[64];
    int count_ = 0;
};

}  // namespace geo
