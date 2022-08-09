import tkinter as tk


def help_tips():
    def tips():
        tip = tk.Tk()
        tip.title('帮助')
        tip.geometry('300x225+500+300')
        t1 = tk.Label(tip, text="1，请先打开系统\n2，注册时请对准人脸，光线充足\n3，训练时禁止进行其他操作")
        t1.place(x=30, y=25)
        # t1 = tk.Label(tip, text="第二")
        # t1.place(x=30, y=50)
        # t1 = tk.Label(tip, text="第三")
        # t1.place(x=30, y=75)
        # bt_sure = tk.Button(tip, text='确定', height=1, width=8)
        # bt_sure.place(x=120, y=180)
        tip.mainloop()
    tips()
