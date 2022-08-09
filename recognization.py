import time
from tkinter import *
import tkinter as tk
import tkinter.messagebox
from tkinter import filedialog  # 文件控件
import pymysql
import threading # 多线程
import help_tip
from late_name import late
from PIL import Image, ImageTk  # 图像控件
import cv2
import cv2
import tkinter as tk
from tkinter import filedialog#文件控件
from PIL import Image, ImageTk#图像控件
import threading#多线程
import precamera
from db import mysql_conn
from getimgdata import GetImgData
# imgs, labels, number_name = GetImgData().readimg()  # 读取数据
# from sklearn.model_selection import train_test_split
# from cnn_net import CnnNet
import time
from db import mysql_conn
window = tk.Tk()
window.title('人脸识别签到系统')
sw = window.winfo_screenwidth()    # 获取屏幕宽
sh = window.winfo_screenheight()   # 获取屏幕高
wx = 600
wh = 800
window.geometry("%dx%d+%d+%d" %(wx,wh,(sw-wx)/2,(sh-wh)/2-100)) # 窗口至指定位置
canvas = tk.Canvas(window,bg='#c4c2c2',height=wh,width=wx)      # 绘制画布
canvas.pack()







#knn
"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.

3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:

$ pip3 install scikit-learn

"""

import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


import time
from tkinter import *
import tkinter as tk
import tkinter.messagebox
from tkinter import filedialog  # 文件控件
import pymysql
import threading # 多线程
import help_tip
from late_name import late
from PIL import Image, ImageTk  # 图像控件
import cv2
import cv2
import tkinter as tk
from tkinter import filedialog#文件控件
from PIL import Image, ImageTk#图像控件
import threading#多线程
import precamera
from getimgdata import GetImgData
# imgs, labels, number_name = GetImgData().readimg()  # 读取数据
# from sklearn.model_selection import train_test_split
# from cnn_net import CnnNet
import time
from db import mysql_conn

def train0(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()











#显示停顿的注册人脸鉴别
flag = 0
use_image = ''
image2 = ''
def cut():
    global  flag

def solidate():
    global  flag,use_image
    def cc():
        while True:
            try:
                if flag == 1:
                    canvas.create_image(270, 560, anchor='nw', image=use_image)

                if image1 != '' :

                    canvas.create_image(270, 560, anchor='nw', image=image1)
                    use_image = image1
            except:
                pass

    # cc()
    t=threading.Thread(target=cc)
    t.start()
solidate()

#---------------打开摄像头获取图片
def video_demo():
    def cc():
        cap = cv2.VideoCapture(0)

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            tt = frame

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            xmlfile = r'haarcascade_frontalface_default.xml'

            face_cascade = cv2.CascadeClassifier(xmlfile)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(5, 5),
            )

            cv2.imwrite(r'tu6.png', frame)
            predictions = predict(r'tu6.png', model_path="trained_knn_model.clf")

            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                print("发现1个目标!".format(len(faces)))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (left, top), (left + (right - top), top + (right - top)), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
                    cur = mysql_conn.cursor()
                    sql_update = "update users set state=1"  + " where user_name='%s" % name + "'"
                    cur.execute(sql_update)
                    mysql_conn.commit()
                    # def dd():
                    T2 = tk.Label(window, text=f"{name}~ 签到成功！！！", font='楷体', bd=14, width=80, fg='black')
                    T2.place(x=0, y=700)
                    time.sleep(0.2)
                    # time.sleep(2)
                    T2.destroy()



                    # h = threading.Thread(dd)
                    # h.start()








                print("- Found {} at ({}, {})".format(name, left, top))
                    # # 左眼
                    # cv2.circle(frame , (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
                    # # 右眼
                    # cv2.circle(frame , (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
                    # # 嘴巴
                    # cv2.rectangle(frame , (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), color)
                try:
                    hh = frame
                    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                    size = 64
                    import  numpy as np
                    need = Image.fromarray(frame)
                    need = need.crop((x, y, w+x, y+h))
                    need = cv2.cvtColor(np.array(need), cv2.COLOR_RGB2BGR)
                    need0 = cv2.cvtColor(np.array(need), cv2.COLOR_RGB2GRAY)

                    face_gray = cv2.resize(need, (size, size))  # 压缩成指定大小
                    face_gray0 = cv2.resize(need0, (size, size))  # 压缩成指定大小

                    global image0
                    global image2
                    image0 = face_gray0
                    image2 = hh
                    face_gray = Image.fromarray(face_gray)

                    # cv2.imwrite(newpath + '/' + str(i) + '.jpg', face_gray)  # 保存检测出的人脸图片 无法读取中文路径
                    # savePath = (newpath + "/%d.jpg" % i)
                    # cv2.imencode('.jpg', face_gray)[1].tofile(savePath)



                    image_file=ImageTk.PhotoImage(face_gray)
                    global  image1
                    image1 = image_file
                    # canvas.create_image(200,  650, anchor='nw', image=image_file)
                    #
                    #暂时搬运主图测试
                    img = Image.fromarray(hh)
                    image_file=ImageTk.PhotoImage(img)

                    canvas.create_image(0,0,anchor='nw',image=image_file)



                except Exception as file:
                    # print(file)

                    img = Image.fromarray(frame)
                    image_file=ImageTk.PhotoImage(img)
                    canvas.create_image(0,0,anchor='nw',image=image_file)
            else:
                img = Image.fromarray(frame)
                image_file = ImageTk.PhotoImage(img)
                canvas.create_image(0, 0, anchor='nw', image=image_file)

    t=threading.Thread(target=cc)
    t.start()
# def camera():
#     t=threading.Thread(target=cc)
#     t.start()

data_list = {}
def register():
    def data():
        try:
            stu_num = E1.get()
            # def cc():
            stu_name = E2.get()
            if stu_num.isnumeric() :

                print('正在注册')
                precamera.pre_image(E2.get(), image2)
                try:
                    cursor = mysql_conn.cursor()
                    sql = f"INSERT INTO users(user_id,user_name,state,userscol) VALUES({stu_num},'{stu_name}', 0,"'null'")"
                    cursor.execute(sql)
                    mysql_conn.commit()
                except:
                    tk.messagebox.showerror(title='警告信息', message='数据库错误')

            else:
                tk.messagebox.showerror(title='警告信息', message='信息格式有误\n请重新输入')
                return

        except EXCEPTION as file:
            print(file)
            print('illegle parameter')
            return

        data_list[stu_num] = stu_name
        print(data_list)
        # try:
        cursor = mysql_conn.cursor()
        sql = "insert into users(user_id, user_name, state) values ('%s', '%s', %s)" % (
            stu_num, stu_name, 0)
        cursor.execute(sql)
        mysql_conn.commit()

        tk.messagebox.showinfo(title='注册信息', message='注册成功，存入数据库成功')
        # except EXCEPTION as file:
        #     print(file)
        #     tk.messagebox.showerror(title='注册信息', message='注册失败，存入数据库失败')


    # def sign_to_website():
    #     stu_num = E1.get()
    #     stu_name= E2.get()
    #     db = pymysql.connect('localhost', 'mysql57', '123456', 'Northwind', charset='utf8') # 数据库信息要改
    #     cursor = db.cursor()
    #     sql_search = 'select * from users where account=%s' % stu_num
    #     if cursor.execute(sql_search):
    #         tk.messagebox.showerror(title='error', message='该用户已经注册')
    #     else:
    #         sql_insert = "insert into users(stu_num,stu_name) values(%s,%s)" % (stu_num, stu_name)
    #         try:
    #             cursor.execute(sql_insert)
    #             db.commit()
    #             tk.messagebox.showinfo('welcome', '注册成功')
    #             sign_up_window.destroy()
    #         except:
    #             db.rollback()
    # sign_up_window = tk.Toplevel(window)
    # sign_up_window.geometry('320x180')
    # sign_up_window.title('注册')
    #
    # stu_num = tk.IntVar()
    # tk.Label(sign_up_window, text='学号:').place(x=30, y=10)
    # entry_new_account = tk.Entry(sign_up_window, textvariable=stu_num)
    # entry_new_account.place(x=130, y=10)
    #
    # stu_name = tk.StringVar()
    # tk.Label(sign_up_window, text='姓名:').place(x=30, y=50)
    # entry_usr_pwd = tk.Entry(sign_up_window, textvariable=stu_name)
    # entry_usr_pwd.place(x=130, y=50)
    #
    # btn_comfirm_sign_up = tk.Button(sign_up_window, text='注册', command=sign_to_website)
    # btn_comfirm_sign_up.place(x=180, y=130)
    #
    r = threading.Thread(target=data)
    r.start()

# def late():
#     def late_list():
#         # Treeview控件
#         list = Tk()
#         list.title('缺勤名单')
#         list.geometry("250x225")# frame容器放置表格
#         frame01 = Frame(list)
#         frame01.place(x = 10,y = 10,width =240,height = 220 )
#         # 加载滚动条
#         scrollBar = Scrollbar(frame01)
#         scrollBar.pack(side = RIGHT,fill = Y)
#         # 准备表格TreeView
#         tree = Treeview(frame01,columns = ("学号","姓名","状态"),show = "headings",yscrollcommand = scrollBar.set)
#         # 设置每一列的宽度和对齐方式
#         tree.column("学号",width = 80,anchor = "center")
#         tree.column("姓名",width = 80,anchor = "center")
#         tree.column("状态",width = 60,anchor = "center")
#
#         # 设置表头的标题文本
#         tree.heading("学号",text = "学号")
#         tree.heading("姓名",text = "姓名")
#         tree.heading("状态",text = "状态")
#
#         # 设置关联
#         scrollBar.config(command = tree.yview)
#         # 加载表格信息
#         tree.pack()
#         # 插入数据(后面改数据库导入未签到学员)
#         for i in range(10):
#         # i 是索引
#           tree.insert("",i,values=["1"+str(i),"林双*","缺勤"])
#     late_list()
def Late():
    late()

x = 500  # 训练的次数，可更改
def train():
    def training():

        time.sleep(2)
        # 设置下载进度条
        tk.Label(window, text='训练进度:', ).place(x=50, y=700)
        canvas = tk.Canvas(window, width=465, height=25, bg="white")
        canvas.place(x=110, y=700)
        print("Training KNN classifier...")
        classifier = train0("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
        print("Training complete!")
        # path = './camera_test_img'  # 灰度图路径
        # imgs, labels, number_name = GetImgData(dir=path).readimg()
        # x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2)
        # # 设置下载进度条
        # tk.Label(window, text='训练进度:', ).place(x=50, y=700)
        # canvas = tk.Canvas(window, width=465, height=25, bg="white")
        # canvas.place(x=110, y=700)
        # cnnNet = CnnNet(modelfile='./temp/train-model',
        #                 imgs=x_train, labels=y_train)
        #
        # train_class = cnnNet.cnnTrain(maxiter=1000,  # 最大迭代次数
        #                               accu=0.99, )  # 指定正确率（499次之后）
        #
        # for index, out in train_class:
        #     print(index, out)
        #
        #     # 填充进度条
        #     global x
        #     fill_line = canvas.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
        #     # n = 500 / x  # 465是矩形填充满的次数
        #     if out%50 == 0:
        #         canvas.coords(fill_line, (0, 0, out, 60))
        #         window.update()

        # 显示下载进度


            # time.sleep(0.02)  # 控制进度条流动的速度
        # 清空进度条
        fill_line = canvas.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
        canvas.coords(fill_line, (0, 0, 500, 60))
        window.update()
        time.sleep(2)
        fill_line = canvas.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="white")
        x = 500  # 未知变量，可更改
        n = 465 / x  # 465是矩形填充满的次数

        for t in range(x):
            n = n + 5 / x
            # 以矩形的长度作为变量值更新
            canvas.coords(fill_line, (0, 0, n, 60))
            window.update()
            time.sleep(0)  # 时间为0，即飞速清空进度条
        tk.messagebox.showinfo(title='训练进度', message='训练完成')
    # p = threading.Thread(target=training)
    training()
    # p.start()


# def help_tips():
#     def tips():
#         tip = tk.Tk()
#         tip.title('帮助')
#         tip.geometry('300x225')
#         t1 = tk.Label(tip, text="第一\n第二\n第三")
#         t1.place(x=30, y=25)
#         # t1 = tk.Label(tip, text="第二")
#         # t1.place(x=30, y=50)
#         # t1 = tk.Label(tip, text="第三")
#         # t1.place(x=30, y=75)
#         # bt_sure = tk.Button(tip, text='确定', height=1, width=8)
#         # bt_sure.place(x=120, y=180)
#         tip.mainloop()
#
#     h = threading.Thread(target=tips)
#     h.start()
def tt():
    h = threading.Thread(target=help_tip.help_tips().tips)
    h.start()




bt_start = tk.Button(window,text='打开系统',height=2,width=15,activeforeground='green',font='楷体',command=video_demo)
bt_start.place(x=30,y=500)


bt_start0 = tk.Button(window,text='注册',height=2,width=15,activeforeground='green',font='楷体',command=register)
bt_start0.place(x=170,y=500)

L1 = tk.Label(window, text="学号",font='楷体')
L1.place(x=30,y=570)
E1 = tk.Entry(window, bd =3,font='楷体')
E1.place(x=70,y=570)
L2 = tk.Label(window, text="姓名",font='楷体')
L2.place(x=30,y=600)
E2 = tk.Entry(window, bd =3,font='楷体')
E2.place(x=70,y=600)


bt_start1 = tk.Button(window,text='查看缺勤名单',height=2,width=15,activeforeground='green',font='楷体',command=Late)
bt_start1.place(x=310,y=500)

bt_start2 = tk.Button(window,text='开始训练',height=2,width=15,activeforeground='green',font='楷体',command=train)
bt_start2.place(x=450,y=500)

# 确定是否使用此照片进行训练
L3 = tk.Label(window, text="是否使用此照片进行训练?",font='楷体')
L3.place(x=380,y=570)
bt_bool1 = tk.Button(window,text='是',height=1,width=6,activeforeground='green',font='楷体',command=train)
bt_bool1.place(x=380,y=600)
bt_bool2 = tk.Button(window,text='选取此图',height=1,width=6,activeforeground='green',font='楷体',command=cut)
bt_bool2.place(x=445,y=600)

bt_help2 = tk.Button(window,text='帮助',height=1,width=6,activeforeground='green',font='楷体',command=tt)
bt_help2.place(x=510,y=600)

window.mainloop()












