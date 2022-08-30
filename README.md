# face_recognization
环境安装：python PyCharm
python库：dlib（Cmake、visual basic、C++依赖环境）
opencv库、tkinter库、pymysql、PIL、threading、time
打开系统按钮：打开摄像头捕获人脸
注册按钮：将人脸、学号、姓名信息存储
查看缺勤名单：将没有识别签到的学生学号姓名导出到打开的名单界面
是否开始训练：点击是按钮进行人脸模型的训练
帮助按钮：查看使用软件的些许注意事项

注意事项：
1，请先打开系统。
2.注册时请对准人脸，光线充足。
3.训练时禁止进行其他操作。

recognition.py 是主文件
db.py是数据库文件
help_tip.py是帮助窗口文件
precamera.py是预处理文件
late_name.py是缺勤名单
trained_knn_model.clf是KNN算法模型
knn_examples文件夹包含了训练集和测试集
仅供学习交流探讨使用 qq 2934218525 备注来意
