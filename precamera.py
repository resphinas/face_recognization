import os
import cv2
import numpy as np
def pre_image(name,image0):
    # global  image0
    print(image0,'d')
    if str(image0) != '':

        filepath = os.path.join('.\\knn_examples\\train', name)  # 路径拼接
        if not os.path.exists(filepath):  # 看是否需要创建路径
            os.makedirs(filepath)
        for i in range(3):  # 开始拍照
            print(i)
            savePath = (filepath + "/%d.jpg" % i)
            # image0 = image0.astype( np.uint8 )
            # image0 = cv2.imdecode(image0, -1)
            # cv2.imwrite(savePath, image0)  # 保存检测出的人脸图片 无法读取中文路径
            cv2.imencode('.jpg', image0)[1].tofile(savePath)
            # img = cv2.cvtColor(np.array(image0), cv2.COLOR_RGB2BGR)  # PIL转cv2
            # cv2.imwrite(savePath,img)
        # image0 = ''
