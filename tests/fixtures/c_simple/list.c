#include "list.h"

#include <stdlib.h>

struct ListNode *list_push(struct ListNode *head, Point value) {
    struct ListNode *node = malloc(sizeof(struct ListNode));
    node->value = value;
    node->next = head;
    return node;
}

void list_free(struct ListNode *head) {
    while (head != NULL) {
        struct ListNode *next = head->next;
        free(head);
        head = next;
    }
}

int list_length(const struct ListNode *head) {
    int count = 0;
    while (head != NULL) {
        count++;
        head = head->next;
    }
    return count;
}
