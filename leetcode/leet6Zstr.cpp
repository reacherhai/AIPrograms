#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    static bool cmp(pair<int,int>a, pair<int,int>b)
    {
        if(b.second!=a.second)
            return b.second > a.second;
        else
            return b.first>a.first;
    }
    string convert(string s, int numRows) {
        int n =  s.length();
        const int N = 10000;
        int row[N] = {0};
        int r = 1;
        int flag = 1;
        vector< pair<int,int> > v;
        for (int i=0;i<n;i++)
        {
            row[i] = r;
            v.push_back(make_pair(i,r));
            r += flag;
            if(r == 1|| r == numRows)
                flag = -1*flag;
        }
        sort(v.begin(),v.end(),cmp);
        string ans;
        for (int i=0;i<v.size();i++)
        {
            ans += s[v[i].first];
        }
        return ans;

    }

};

int main()
{
    int n = 5,numRows = 3;
    Solution s;
    string str = s.convert("Apalindrome,",2);
    cout<<str;

}
