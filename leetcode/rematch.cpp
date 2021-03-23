#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    static bool bfs(vector<int>edge[31][26],int start,int endp,string s)
    {
        int q[10000]={0};
        int vis[10000]={0};
        int head = 0;
        int rear = 1;
        q[0] = start;
        vis[start] = 1;
        int m = s.length();
        //必须完整走完整个s，才能判断是否符合。
        //每次应该对q里面的所有起点，都遍历，看能不能通过进来的字符去到新的位置。
        for (int i=0;i<m;i++)
        {
            int temphead = head, temprear = rear;
            for(int j = temphead;j<temprear;j++)
            {
                int temp = q[j];
                vis[temp] = 0;
                for (int k=0;k<edge[temp][s[i]-'a'].size();k++)
                {
                    int x = edge[temp][s[i]-'a'][k];
                    if(!vis[x])
                    {
                        vis[x] = 1;
                        q[rear++] = x;
                    }
                }
            }
            head = temprear;
        }
        for (int i=0;i<rear;i++)
            cout<<q[i]<<" ";
        cout<<endl;

        for (int i=head;i<rear;i++)
        {
            int x = q[i];
            if(x == endp)
                return true;
        }
        return false;
    }
    bool isMatch(string s, string p) {
        const int N = 31;
        vector<int>edge[N][26];
        int m = s.length(),n = p.length();
        //status represents the end point.
        int status = 0;
        for(int i=0;i<n;i++)
        {
            if(p[i]>='a'&&p[i]<='z'&&p[i+1]!='*')
            {
                status ++;
                edge[status-1][p[i]-'a'].push_back(status);
            }
            if(p[i]>='a'&&p[i]<='z'&&p[i+1]=='*')
            {
                edge[status][p[i]-'a'].push_back(status);
            }
            if(p[i]=='.'&&p[i+1]=='*')
            {
                for (int j=0;j<26;j++)
                    edge[status][j].push_back(status);
            }
            if(p[i]=='.')
            {
                status++;
                for (int j=0;j<26;j++)
                    edge[status-1][j].push_back(status);
            }
        }
        if(bfs(edge,0,status,s))
            return true;
        else
            return false;
    }
};


int main()
{
    Solution s;
    cout<<s.isMatch("aa","a");
}
