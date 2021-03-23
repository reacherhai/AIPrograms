#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int myAtoi(string s) {
        int i = 0;
        while (s[i]==' ')
            i++;
        int flag = 1;
        int ans = 0;
        if(s[i]=='-')
        {
            i++;
            flag = -1;
        }
        else if(s[i]=='+')
        {
            i++;
            flag = 1;
        }
        else if(!(s[i]>='0'&&s[i]<='9') )
            return 0;
        while(s[i]>='0'&&s[i]<='9')
        {
            int res = s[i]-'0';
            if(flag * ans>INT_MAX/10 || (flag * ans==INT_MAX/10&&res>7) )
                return INT_MAX;
            if(flag * ans<INT_MIN/10 || (flag * ans==INT_MIN/10&&res<=-8) )
                return INT_MIN;
            ans = ans*10 +res;
            i++;
        }
        ans = ans * flag;
        return ans;
    }
};

int main()
{
    string s = "           22122222223    ";
    Solution ss;
    while (!s.empty())
    {
        cin>>s;
        cout<<ss.myAtoi(s)<<endl;
    }
}
