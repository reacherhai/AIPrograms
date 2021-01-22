#include <bits/stdc++.h>
#include <windows.h>
using namespace std;

int puzzle[10][10]={0};//存储每个位置实际的值
int values[10][10][10]={1};//存储每个位置可能的值，初始化时全部可能为1，不可能则为0.
int constraints[10][10][8]={0};//0-7 8 种约束条件
//0-7分别小于左，大于左，小于右，大于右，小于上，大于上，小于下，大于下
int consnum[8] = {3,2,1,0,7,6,5,4};
int row, col;
int n;

//更新节点的函数，包含对节点(x,y)赋值与对所有与其相关的约束条件更新
void updateNode(int value,int x,int y)
{
    puzzle[x][y] = value;
    for (int i=1;i<=col;i++)
    {
        values[x][i][value] = 0;
    }
    for (int i=1;i<=row;i++)
    {
        values[i][y][value] = 0;
    }
    //上两句对行列alldif进行更新，同行列不能与其相等。
    values[x][y][value] = 1;
    for (int i=0;i<8;i++)
    {
        //对八种约束条件进行检查并更新周围节点的取值范围
        if(constraints[x][y][i]==1)
        {
            switch(i)
            {
            case 0:
                for (int i=1;i<=value;i++)
                    values[x][y-1][i] = 0 ;
                break;
            case 1:
                for (int i=value;i<=n;i++)
                    values[x][y-1][i] = 0 ;
                break;
            case 2:
                for (int i=1;i<=value;i++)
                    values[x][y+1][i] = 0 ;
                break;
            case 3:
                for (int i=value;i<=n;i++)
                    values[x][y+1][i] = 0 ;
                break;
            case 4:
                for (int i=1;i<=value;i++)
                    values[x-1][y][i] = 0 ;
                break;
            case 5:
                for (int i=value;i<=n;i++)
                    values[x-1][y][i] = 0 ;
                break;
            case 6:
                for (int i=1;i<=value;i++)
                    values[x+1][y][i] = 0 ;
                break;
            case 7:
                for (int i=value;i<=n;i++)
                    values[x+1][y][i] = 0 ;
                break;
            }
        }
    }
}

//判定终止状态
int allAssigned()
{
    for (int i=1;i<=row;i++)
        for (int j=1;j<=col;j++)
    {
        if(!puzzle[i][j])
            return 0;
    }
    return 1;
}
//输出values数组
void printval()
{
    for (int i=1;i<=row;i++){
        for (int j=1;j<=col;j++){
            cout<<i<<","<<j<<":"<<endl;
            for(int k=1;k<=n;k++)
            {
                if(values[i][j][k])
                    cout<<k<<" ";
            }
            cout<<endl;
        }
    }
}

//用MRV的策略选择一个index
int MRVPick()
{
    int minnum = 10;
    int minIndex = 0;
    for (int i=1;i<=row;i++)
        for (int j=1;j<=col;j++)
    {
        int count = 0;
            for (int k=1;k<=n;k++)
                if(values[i][j][k])
                    count++;
        if(count<minnum & !puzzle[i][j])
        {
            minnum = count;
            minIndex = (i-1)*n + j;
        }
    }
    return minIndex;
}

int DWOcheck(int x, int y)
{
    for (int i=1;i<=n;i++)
    {
        if(values[x][y][i]!=0)
            return 0;
    }
    return 1;
}

bool in(tuple <int,int, int > tt, queue<tuple<int,int,int> > q)
{
    int size = q.size();
    for (int i=0;i<size;i++)
    {
        tuple <int,int,int>temp = q.front();
        q.pop();
        if(temp == tt )
            return true;
    }
    return false;
}

int puzzleTest[10][10];

bool rowDfs(int r,int level,int levelnow)
{
    if(level == n+1)
    {
        return true;
    }
    if(level==levelnow)
        rowDfs(r,level+1,levelnow);
    for (int val=1;val<=n;val++)
    {
        if(values[r][level][val])
        {
            int flag = 1;
            for (int j=1;j<=level-1;j++)
            {
                if(val==puzzleTest[r][j])
                    flag=0;
            }
            if(flag)
            {
                puzzleTest[r][level] = val;
                if(rowDfs(r,level+1,levelnow))
                    return true;
            }
        }
    }
    return false;
}

bool colDfs(int c,int level,int levelnow)
{
    if(level == n+1)
        return true;
    if(level==levelnow)
        colDfs(c,level+1,levelnow);
    for (int val=1;val<=n;val++)
    {
        if(values[level][c][val])
        {
            int flag = 1;
            for (int j=1;j<=level-1;j++)
            {
                if(val==puzzleTest[j][c])
                    flag=0;
            }
            if(flag)
            {
                puzzleTest[c][level] = val;
                if(colDfs(c,level+1,levelnow))
                    return true;
            }
        }
    }
    return false;
}

queue <tuple<int,int,int> > GACqueue;
queue <tuple<int,int,int> > tempqueue;

void printqueue(queue <tuple<int,int,int> > q)
{
    int size = q.size();
    for (int i=0;i<size;i++)
    {
        tuple <int,int,int>temp = q.front();
        q.pop();
        cout<< get<0> (temp)<<" "<<get<1> (temp)<<" "<<get<2> (temp)<<endl;
    }
}

void pushAllConstraints(int Vx,int Vy,int x,int y)
{
    for (int i=0;i<8;i++)
        {
            if(constraints[Vx][Vy][i] && !in(std::make_tuple(x,y,consnum[i]),GACqueue) )
                {
                    tuple <int,int,int> tt = std::make_tuple(Vx,Vy,i);
                    if(!in(tt, GACqueue))
                    {
                        //tempqueue.push(tt);
                        GACqueue.push(tt);
                    }
                }
        }

    tuple <int,int,int> tt1 = std::make_tuple(-2,-2,Vy);
    if(!in(tt1, GACqueue))
    {
        //tempqueue.push(tt1);
        GACqueue.push(tt1);
    }
    tuple <int,int,int> tt2 = std::make_tuple(-1,-1,Vx);
    if(!in(tt2, GACqueue))
    {
        //tempqueue.push(tt2);
        GACqueue.push(tt2);
    }

}

int judge(int x,int y,int Vx,int Vy,int cnum)
{
    int flag = 0 ;
    int minvalue = 100;
    int maxvalue = -1;
    for (int i=1;i<=n;i++)
    {
        if(values[Vx][Vy][i])
        {
            minvalue = min(minvalue,i);
            maxvalue = max(maxvalue,i);
        }
    }
    //cout<<"domain delete:";
    for (int i=1;i<=n;i++)
    {
        flag = 0;
        if(values[x][y][i])
        {
            if(cnum%2 ==0 ) // 2的倍数为小于
            {
                if(i<maxvalue)
                    flag = 1;
            }
            else // 奇数约束为大于
            {
                if(i>minvalue)
                    flag = 1;
            }
            if(!flag)
            {
                values[x][y][i] = 0 ;
                //cout<<x<<" "<<y<<" "<<i<<endl;
                if(DWOcheck(x,y))
                    {
                        GACqueue = queue <tuple<int,int,int> > ();
                    }
                else
                    pushAllConstraints(x,y,Vx,Vy);
            }
        }
    }
    return 1;
}

int GAC_enforce()
{
    //tempqueue = GACqueue;
    while (!GACqueue.empty())
    {
        //cout<<"当前队列："<<endl;
        //printqueue(GACqueue);
        //cout<<endl;

        tuple <int,int,int> temp = GACqueue.front();
        GACqueue.pop();
        int cNum = get<2> (temp), x = get<0> (temp) , y = get<1> (temp);
        //cout<<"正在处理:"<<x<<y<<cNum<<endl;
        int Vx ,Vy ;
        if(x>0&&y>0)// constraint is about direction
        {
            switch(cNum)
                {
                case 0:
                    Vx = x;
                    Vy = y-1;
                    /*
                    for (int i=1;i<=n;i++)
                        if(values[x][y-1][i]&& i>minvalue)
                            flag = 1;
                    */
                    break;
                case 1:
                    Vx = x;
                    Vy = y-1;
                    /*
                    for (int i=1;i<=n;i++)
                        if(values[x][y-1][i]&& i<maxvalue)
                            flag = 1;
                    */
                    break;
                case 2:
                    Vx = x;
                    Vy = y+1;
                    break;
                case 3:
                    Vx = x;
                    Vy = y+1;
                    break;
                case 4:
                    Vx = x-1;
                    Vy = y;
                    break;
                case 5:
                    Vx = x-1;
                    Vy = y;
                    break;
                case 6:
                    Vx = x+1;
                    Vy = y;
                    break;
                case 7:
                    Vx = x+1;
                    Vy = y;
                    break;
                }
            if(!judge(x,y,Vx,Vy,cNum) || !judge(Vx,Vy,x,y,consnum[cNum]))
                return 0;
        }
        //cout<<"domain deletes";
        memset(puzzleTest,0,sizeof(puzzleTest));
        if(x == -1 && y == -1)  // constraint is about row
        {
            int nrow = cNum;
            for (int i=1; i<=n ; i++)
            {
                for (int val=1;val<=n;val++)
                    if(values[nrow][i][val])
                    {
                        puzzleTest[nrow][i] = val;
                        if(!rowDfs(nrow,1,i))
                        {
                            //cout<<"values:"<<endl;
                            //printval();
                            values[nrow][i][val] = 0;
                            //cout<<nrow<<" "<<i<<" "<<val<<endl;
                            if(DWOcheck(nrow,i))
                            {
                                GACqueue = queue <tuple<int,int,int> > ();
                                return 0;
                            }
                            else
                            {
                                pushAllConstraints(nrow,i,-1,-1);
                            }

                        }
                        puzzleTest[nrow][i] = 0;
                    }
            }
        }
        if(x == -2 && y == -2) // constraint is about col
        {
            int ncol = cNum;
            for (int i=1; i<=n ; i++)
            {
                for (int val=1;val<=n;val++)
                    if(values[i][ncol][val])
                    {
                        if(!colDfs(ncol,1,i))
                        {
                            //cout<<"values:"<<endl;
                            //printval();
                            values[i][ncol][val] = 0;
                            //cout<<i<<" "<<ncol<<" "<<val<<endl;
                            if(DWOcheck(i,ncol))
                            {
                                GACqueue = queue <tuple<int,int,int> > ();
                                return 0;
                            }
                            else
                                pushAllConstraints(i,ncol,-2,-2);
                        }
                    }
            }
        }
        //cout<<"after pushing constraints:"<<endl;
        //printqueue(GACqueue);
    }
    return 1;

    //cout<<"剔除后的value"<<endl;
    //printval();
    //cout<<endl;
}

void GAC(int level)
{
    if(allAssigned())
    {
        for (int i=1;i<=row;i++){
            for (int j=1;j<=col;j++)
                cout<<puzzle[i][j]<<" ";
            cout<<endl;
        }
        exit(0);
    }
    //使用一个栈内的小数组记录对应当时状态的取值范围。
    int oldValues[10][10][10];
    memcpy(oldValues,values,sizeof(values));
    int MinIndex = MRVPick();
    if(MinIndex == 0)
        return;
    //获取MRV得到的行与列
    int c = (MinIndex-1) % n + 1;
    int r = (MinIndex-1) / n + 1;
    //接下来对每一个该位置可能的取值分析
    for (int i=1;i<=n;i++)
        if(values[r][c][i])
    {
        int d = i;
        //使用这个值更新节点
        //cout<<"当前节点"<<endl;
        updateNode(d,r,c);
        //cout<<r<<" "<<c<<" "<<d<<endl;
        //接下来作GACcheck。
        for (int j=0;j<8;j++)
        {
            if(constraints[r][c][j])
            {
                tuple <int,int,int>t = std::make_tuple(r,c,j);
                GACqueue.push(t);
            }
        }
        tuple <int,int,int>tr = std::make_tuple (-1,-1,r);
        tuple <int,int,int>tc = std::make_tuple (-2,-2,c);
        GACqueue.push(tr);
        GACqueue.push(tc);

        //如果无事发生

        //cout<<"当前values"<<endl;
        //printval();
        //cout<<endl;
        if(GAC_enforce())
        {
            //cout<<"the point " << r<<","<<c<<" is chosen "<<"as "<< d<<endl;
            GAC(level + 1);
        }
        //最后恢复到遍历此节点前的状态，恢复剪枝。
        memcpy(values,oldValues,sizeof(oldValues));
    }
    puzzle[r][c] = 0;
    //最后要把这个点的值取消。因为已遍历所有可能
    return;

}

int main()
{
    cin>>n;
    row = col = n;
    for (int i=1;i<10;i++)
        for(int j=1;j<10;j++)
            for (int k=1;k<10;k++)
                values[i][j][k]=1;

    int numC = 0;
    cin>>numC;
    for (int i=0;i<2*numC;i++)
    {
        int x,y,t;
        cin>>x>>y>>t;
        constraints[x][y][t] = 1;

    }
    for (int i=1;i<=row;i++)
        for (int j=1;j<=col;j++)
    {
        int value;
        cin>>value;
        if(value)
            updateNode(value,i,j);
    }
    cout<<endl;
    GAC(0);
    Sleep(1000000);

}
