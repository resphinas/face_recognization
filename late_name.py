from tkinter import *
from tkinter.ttk import *
from db import mysql_conn



def late():
    def late_list():
        cursor = mysql_conn.cursor()
        a = cursor.execute("select * from users where state=0")
        a = cursor.fetchall()

        # Treeview控件
        list = Tk()
        list.title('缺勤名单')
        list.geometry("250x225")# frame容器放置表格
        frame01 = Frame(list)
        frame01.place(x = 10,y = 10,width =240,height = 220 )
        # 加载滚动条
        scrollBar = Scrollbar(frame01)
        scrollBar.pack(side = RIGHT,fill = Y)
        # 准备表格TreeView
        tree = Treeview(frame01,columns = ("学号","姓名","状态"),show = "headings",yscrollcommand = scrollBar.set)
        # 设置每一列的宽度和对齐方式
        tree.column("学号",width = 80,anchor = "center")
        tree.column("姓名",width = 80,anchor = "center")
        tree.column("状态",width = 60,anchor = "center")

        # 设置表头的标题文本
        tree.heading("学号",text = "学号")
        tree.heading("姓名",text = "姓名")
        tree.heading("状态",text = "状态")

        # 设置关联
        scrollBar.config(command = tree.yview)
        # 加载表格信息
        tree.pack()
        # 插入数据(后面改数据库导入未签到学员)
        for i in range(len(a)):
        # i 是索引

          tree.insert("",i,values=["1"+str(a[i][0]),a[i][1],'缺勤'])
    late_list()