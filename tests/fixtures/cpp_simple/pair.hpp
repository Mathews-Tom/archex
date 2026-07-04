#pragma once

namespace geo {

template <typename T, typename U = T>
class Pair {
public:
    Pair(T first, U second) : first_(first), second_(second) {}

    T getFirst() const {
        return first_;
    }

    U getSecond() const {
        return second_;
    }

private:
    T first_;
    U second_;
};

template <>
class Pair<int> {
public:
    explicit Pair(int value) : value_(value) {}

    int sum() const {
        return value_ + value_;
    }

private:
    int value_;
};

}  // namespace geo
