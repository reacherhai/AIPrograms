#include <bits/stdc++.h>

using namespace std;


class Solution {
public:
    string longestPalindrome(string s) {
        int rk = 1;
        int n = s.length();
        string ans;
        string temp = "";
        int maxl = 0;
        const int N = 1001;
        int dp[N][N]= {0};
        //本题使用动态规划解决
        for (int l = 0;l < n;l++)
            for (int i=0;i+l<n;i++)
        {
            int j = i+l;
            if(l==0)
                dp[i][j] = 1;
            if(l==1)
                dp[i][j] = (s[i]==s[j]);
            else
                dp[i][j] = dp[i+1][j-1] && (s[i]==s[j]);
            if(dp[i][j]&&l+1>maxl)
            {
                maxl = l+1;
                ans = s.substr(i,maxl);
            }
        }
        return ans;
    }
};
