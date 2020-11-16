from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import morphology
import copy
import datetime


class PipeFlowDetect:
    """
    使用图像预处理来过滤噪音
    使用霍夫变换检测圆管
    """
    def __init__(self):
        # 存储输入原图
        self.input_img=None
        # 存储预处理后图像
        self.preprocess_img=None
        # 处理检测到空洞后的图像
        self.predict_hole_img=None
        # 检测到空洞的位置
        self.hole_position=[]


    def fun3(self):
        img = cv2.imread('./pipe/1.png')
        GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, GrayImage = cv2.threshold(GrayImage, 40, 255, 1)
        cv2.imshow('thresh image',GrayImage)
        GrayImage = cv2.medianBlur(GrayImage, 5)
        ret, th1 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(GrayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 3, 5)
        th3 = cv2.adaptiveThreshold(GrayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 3, 5)

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(th2, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        imgray = cv2.Canny(erosion, 30, 100)

        circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 150,
                                   param1=100, param2=20, minRadius=20, maxRadius=100)
        if circles is None:
            print('没有管道口被检测到')
        else:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
            print(len(circles[0, :]))

            cv2.imshow('detected circles', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def read_video(self,video_path='./pipe/pipe_flow2.mp4',detect_flow=False):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 设置保存图片格式
        # out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.avi', fourcc, 10.0,
        out = cv2.VideoWriter('outhole_dark' + '.avi', fourcc, 25.0,
                              (480, 270))  # 分辨率要和原视频对应
        while True:
            flag,img = cap.read()

            if not flag:
                print('video is over')
                break
            print('img shape', img.shape)
            obj.smoth_gray(img_path=img,show=False)
            obj.predict_hole(out)

        cap.release()
        cv2.destroyAllWindows()

    def smoth_gray(self,img_path='./pipe/1.png',show=False):
        if isinstance(img_path,str):
            #传入图像地址
            img = cv2.imread(img_path)
        else:
            # 直接传入了图像
            img = copy.deepcopy(img_path)
        self.input_img = copy.deepcopy(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary_img = cv2.threshold(gray_img, 40, 255, 1)
        arr = binary_img > 0
        cleaned = morphology.remove_small_objects(arr, min_size=50)
        cleaned = morphology.remove_small_holes(cleaned, 50)
        # dst_gray = morphology.remove_small_objects(binary_img, min_size=300, connectivity=1)
        cleaned = (cleaned*255).astype('uint8')
        self.preprocess_img=copy.deepcopy(cleaned)
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(img, cmap='gray')
        # axs[0].set_title('img')
        # axs[1].imshow(cleaned, cmap='gray')
        # axs[1].set_title('cleaned')
        # plt.show()
        if show:
            cv2.imshow('binary_img', binary_img)
            cv2.imshow('dst_gray', cleaned)
            cv2.waitKey(1)


    def predict_hole(self,out):
        if self.preprocess_img is None:
            print('请先预处理图像')
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(self.preprocess_img, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        imgray = cv2.Canny(erosion, 30, 100)

        circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 200,
                                   param1=100, param2=8, minRadius=6, maxRadius=200)
        if circles is None:
            print('没有管道口被检测到')
        else:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(self.input_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(self.input_img, (i[0], i[1]), 2, (0, 0, 255), 3)
                print((i[0], i[1]), i[2])
                # self.hole_position.append([(i[0], i[1]), i[2]])
            print(len(circles[0, :]))

            cv2.imshow('detected circles', self.input_img)
        cv2.imshow('preprocess_img', self.preprocess_img)
        out.write(self.preprocess_img)
        cv2.waitKey(1)

    def detect_flow(self):
        """
        检测流水
        """
        # 方式一 检测洞口区域是否有流水矩形区域
        colour = ((0, 205, 205), (154, 250, 0), (34, 34, 178), (211, 0, 148), (255, 118, 72), (137, 137, 139))  # 定义矩形颜色
        cap = cv2.VideoCapture("./pipe/pipe_flow2.mp4")  # 参数为0是打开摄像头，文件名是打开视频
        fgbg = cv2.createBackgroundSubtractorMOG2()  # 混合高斯背景建模算法

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 设置保存图片格式
        # out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.avi', fourcc, 10.0,
        out = cv2.VideoWriter('outflow' +'.avi', fourcc, 10.0,
                              (100, 120))  # 分辨率要和原视频对应
        cor_num_list=[]
        while True:
            flag, frame = cap.read()  # 读取图片
            if flag is False:
                print('video is over')
                break

            _,frame = cv2.threshold(frame, 180, 255, 0)
            frame = frame[80:200,28:128,:]  #管道口1位置
            print('frame shape',frame.shape)
            # frame = frame[0:80,318:418,:]     #管道口2位置
            fgmask = fgbg.apply(frame)

            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 形态学去噪
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)  # 开运算去噪

            contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找前景

            count = 0
            # 检测到角点个数
            cor_num=0
            for cont in contours:
                Area = cv2.contourArea(cont)  # 计算轮廓面积
                cor_num+=1
                if Area < 10:  # 过滤面积小于10的形状
                    continue

                count += 1  # 计数加一

                print("{}-prospect:{}".format(count, Area), end="  ")  # 打印出每个前景的面积

                rect = cv2.boundingRect(cont)  # 提取矩形坐标

                print("x:{} y:{}".format(rect[0], rect[1]))  # 打印坐标

                cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), colour[count % 6],
                              1)  # 原图上绘制矩形
                cv2.rectangle(frame, (50, 80), (100 , 140), colour[count % 6],
                              1)  # 原图上水管
                cv2.rectangle(fgmask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0xff, 0xff, 0xff),
                              1)  # 黑白前景上绘制矩形

                y = 10 if rect[1] < 10 else rect[1]  # 防止编号到图片之外
                cv2.putText(frame, str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)  # 在前景上写上编号

            cor_num_list.append(cor_num)
            cv2.putText(frame, "count:%d"%(cor_num), (5, 5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)  # 显示总数

            cv2.putText(frame, str(count), (75, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
            print("----------------------------")
            print('cor_num',cor_num)
            cv2.imshow('frame', frame)  # 在原图上标注
            cv2.imshow('frame2', fgmask)  # 以黑白的形式显示前景和背景
            out.write(frame)
            k = cv2.waitKey(30) & 0xff  # 按esc退出
            if k == 27:
                break

        print('均值和方差',np.mean(cor_num_list),np.var(cor_num_list))
        print('长度和所有值',len(cor_num_list),cor_num_list)
        out.release()  # 释放文件


if __name__ == '__main__':
    obj = PipeFlowDetect()
    # 处理视频检测洞口
    # while True:
    # obj.read_video(detect_flow=True)

    # 处理单张图图片检测洞口
    # obj.smoth_gray(img_path='./pipe/1.png')
    # obj.predict_hole()

    # 通过流动时候的运动前景图判断是否有水流
    obj.detect_flow()

