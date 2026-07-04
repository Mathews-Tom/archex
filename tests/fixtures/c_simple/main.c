#include "list.h"
#include "platform.h"
#include "point.h"

#include <stdio.h>

int main(void) {
    Point origin = point_make(0, 0);
    Point other = point_make(3, 4);
    printf("distance^2 = %d\n", point_distance_squared(&origin, &other));

    struct ListNode *head = NULL;
    head = list_push(head, origin);
    head = list_push(head, other);
    printf("length = %d\n", list_length(head));
    list_free(head);

    printf("platform = %s\n", platform_name());
    return 0;
}
