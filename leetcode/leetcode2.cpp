/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *head = nullptr;
        ListNode *tail = nullptr;
        int carry = 0;
        while (l1||l2)
        {
            int n1=0,n2=0;
            if(l1)
                n1 = l1->val;
            if(l2)
                n2 = l2->val;
            int sum = n1 + n2 + carry;
            if(!head)
            {
                head = tail = new ListNode(sum%10);
            }
            else
            {
                tail->next = new ListNode(sum%10);
                tail = tail->next;
            }
            carry  = sum / 10;
            if(l1)
                l1 = l1->next;
            if(l2)
                l2 = l2->next;


        }
        if(carry>0)
            tail->next = new ListNode(carry);
        return head;
    }
};
