#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool isPalindrome(int x) {
        stringstream ss;
        ss<<x;
        string s = ss.str();
        int n = s.length();
        for (int i=0;i<n;i++)
            if(s[i]!=s[n-i-1])
                return false;
        return true;
    }
};

int main()
{
    Solution s;
    queue<int>q;
    while(1)
    {
        int i;
        cin>> i;
        cout<<s.isPalindrome(i);
    }

}
