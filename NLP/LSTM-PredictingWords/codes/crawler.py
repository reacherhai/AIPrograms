# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup as BS
import lxml
import requests
import numpy as np
import pandas as pd
from threading import  Thread, Semaphore
from queue import  Queue
import traceback
import time
import lxml
import re
import sqlite3
from urllib import  parse
import tldextract

sem=Semaphore(12)

class worker(Thread):
    def __init__(self, url, queue):
        super(worker, self).__init__()
        self.url = url
        self.q = queue
        
    def run(self):
        self.crawlPage()
        
    def request(self, url):
        print(url)
        send_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
            "Connection": "keep-alive",
            "Accept-Language": "zh-CN,zh;q=0.8"}
        try:
            html=requests.get(url, timeout=2, stream=False,headers = send_headers)
        except Exception:
                #print('Could not open %s'%url)
            pass
        else:
            return html
        
    def crawlPage(self):
        with sem:
            respond=self.request(self.url)
            if respond==None:
                return None

            respond.encoding='utf-8'
            soup=BS(respond.text,'lxml')
            [x.extract() for x in soup.find_all('script')]
            [x.extract() for x in soup.find_all('style')]
            #排除script和样式标签
            
            title=soup.find('title')
            if title==None:
                title=''
            else:
                title=title.string
            text=soup.get_text()
            
            ignoreFiles =('.ppt','.pptx','.bmp','.jpg','.png','.tif','.gif','.pcx','.tga','.exif','.fpx','.svg','.psd','.cdr','.pcd','.dxf','.ufo','.eps','.ai','.raw','.WMF','.webp','.doc','.docx','.txt','.pdf','.xlsx','.xls','.csv','.rar','.zip','.arj','.gz','.z','.wav','.aif','.au','.mp3','.ram','.wma','.mmf','.amr','.aac','.flac','.avi','.mpg','.mov','.swf','.int','.sys','.dll','.adt','.map','.bak','.bat','.cmd')

            newpages=set()
            links=soup('a')
            if(soup.find("div", class_="article")):
                text = soup.find("div", class_="article").get_text()
            for link in links:
                if ('href' in link.attrs):
                    url=parse.urljoin(self.url,link['href'])
                    if url.endswith(ignoreFiles) or not re.match(r'https://tech.sina.com.cn/.*/2019-\d{2}-\d{2}/*',url):
                        continue
                    if url.find("'")!=-1:continue
                    url=url.split('#')[0]
                    if url[0:4]=='http':
                        newpages.add(url)
            self.q.put((title, self.url, text, newpages))



class crawler:
    def __init__(self, dbname):
        self.con=sqlite3.connect(dbname,check_same_thread = False)
        
    def __del__(self):
        self.con.close()
        
    def dbcommit(self):
        self.con.commit()

    def isindexed(self,url):
        u = self.con.execute("select rowid from urllist where url = '%s'" % url).fetchone()
        if u!=None:
            return True
            #v =self.con.execute("select * from wordlocation where urlid = %d" %u[0]).fetchone()
            #if v!=None: return True
        return False

    def transferContent(self, content):
        return content.replace("'","\\")

    def addtoindex(self, url,title , text):
        if self.isindexed(url): return
        self.con.execute('insert into urllist values(?, ?, ?)',(title,url,self.transferContent(text)) )

    def crawl(self, pages,depth=5):
        for i in range(depth):
            q=Queue()
            newpages=set()
            thread_list=[]
            for page in pages:
                p=worker(page,q)
                p.start()
                thread_list.append(p)

            #分多线程开始爬取，并将爬到的东西都加入到队列中。
            
            for thread in thread_list:
                thread.join()
            #print('read finish')
            
            while not q.empty():
                (title,url,text,url_list)=q.get()
                #print('inserting '+url)
                #单线程地插入数据库。防止出现rc问题。
                self.addtoindex(url,title,text)
                for newurl in url_list:
                    #print(k)
                    if not self.isindexed(newurl):
                        newpages.add(newurl)

            self.dbcommit()            
            #print('commit finish')
            pages=newpages
            
    def createindextables(self):
        num =self.con.execute("select count(*) from sqlite_master where type = 'table' and name = 'urllist'").fetchone()
        if(num[0]==0):
            self.con.execute('create table urllist(title, url primary key, text)')
            self.dbcommit()
        
        
class printer:
    def __init__(self,dbname):
        self.con=sqlite3.connect(dbname)
    
    def __del__(self):
        self.con.close()
        
    def query(self, field, table, clause):
        cur=self.con.execute('select %s from %s where %s'%(field, table, clause))
        rows=[row for row in cur]
        return rows

    
if __name__=="__main__":
    #get namelist and pagelist
    pagelist = []
    namelist = ["腾讯科技","新浪科技","网易科技"]

    pagelist = ["https://new.qq.com/ch/tech/","https://tech.sina.com.cn/","https://news.163.com/"]

    location = "data2/1.新浪新闻.db"
    cler = crawler(location)
    cler.createindextables()
    print ([pagelist[1]])
    cler.crawl([pagelist[1]],10)

    location = 'data2/1.新浪新闻.db'
    pter = printer(location)
    listOfTitle = pter.query('title', 'urllist', '0=0')
    listOfText = pter.query('text','urllist','0=0')


    for i in range(10,1010):
        print (listOfTitle[i][0])
        with open("text2/"+str(i-9)+".txt",'w',encoding= "utf-8") as wf:
            wf.write(listOfText[i][0])

