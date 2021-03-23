#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size();
        int n = nums2.size();
        const int D = 1000000;
        for (int i=0;i<10;i++)
            nums1.push_back(D);
        for (int i=0;i<10;i++)
            nums2.push_back(D);
        int p1 = -1;
        int p2 = -1;
        //不用讨论只有很少数的情况：只要把少数的那个vector填充正无穷即可。
        int sum = 0;
        if( (m + n)%2 == 1 )
        {
            sum = p1 + p2 + 2;
            int flag = 0;
            while(sum != (m+n+1)/2)
            {
                if(nums1[p1+1]<nums2[p2+1])
                    {
                        p1++;
                        flag = 1;
                    }
                else
                {
                    p2++;
                    flag = 0;
                }
                sum++;
            }
            if(flag)
                return nums1[p1];
            else
                return nums2[p2];
        }
        else
        {
            sum = p1 + p2 +2;
            int flag =0;
            while(sum != (m+n)/2)
            {
                if(nums1[p1+1]<nums2[p2+1])
                {
                    p1++;
                    flag = 1 ;
                }
                else
                {
                    p2++;
                    flag = 0;
                }
                sum++;
            }
            int a1 = 0;
            if(flag)
                a1 = nums1[p1];
            else
                a1 = nums2[p2];
            if(nums1[p1+1]<nums2[p2+1])
                return (a1+ nums1[p1+1])/2;
            else
                return (a1 +nums2[p2+1])/2;
        }
    }
};

int main()
{
}
