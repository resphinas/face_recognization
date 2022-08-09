
import pymysql

# 连接数据库
try:
    mysql_conn = pymysql.Connection(
        host='localhost',  # 主机地址
        port=3307,  # 端口号
        user='root',  # 登录用户名
        password='4754604072',  # 登录密码
        database='data0',  # 连接的数据库名称
        charset='utf8',  # utf-8的编码
    )
except:
    mysql_conn = None

