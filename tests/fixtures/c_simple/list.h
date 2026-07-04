#ifndef LIST_H
#define LIST_H

#include "point.h"

struct ListNode {
    Point value;
    struct ListNode *next;
};

struct ListNode *list_push(struct ListNode *head, Point value);
void list_free(struct ListNode *head);
int list_length(const struct ListNode *head);

#endif /* LIST_H */
