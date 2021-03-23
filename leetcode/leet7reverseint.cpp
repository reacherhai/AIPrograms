#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    static int power(int x,int y)
    {
        int ans = 1;
        for (int i=0;i<y;i++)
            ans*=x;
        return ans;
    }
    int reverse(int x) {
        int ans = 0;
        int q = x;
        const int N = 33;
        int nums[N]={0};
        int pos = 1;
        while (q!=0)
        {
            int num = q%10;
            q = q/10;
            if(ans>INT_MAX/10 || (ans == INT_MAX/10&&num>7))
                return 0;
            if(ans<INT_MIN/10 || (ans == INT_MIN/10&&num<-8))
                return 0;
            ans =  ans*10 + num;
        }
        return ans;
    }
};


int main()
{
    int c=1;
    Solution s;
    while(c)
    {
        cin>>c;
        cout<<s.reverse(c)<<endl;
    }
}
