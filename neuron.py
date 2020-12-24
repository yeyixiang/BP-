
import numpy as np
import pandas as pd
from plot import plot_number
from scipy.special import expit
from tkinter import ttk
import tkinter as tk
from plot import plot_errors
from matplotlib import pyplot as plt
import cv2
import tkinter.messagebox as tkmegbox
class Neuron:
    """神经网络"""

    def __init__(self, inp_num, hid_num, out_num):
        """初始化神经网络"""

        # 初始化输入结点数、隐藏层结点数，输出结点数
        self.inp_num = inp_num
        self.hid_num = hid_num
        self.out_num = out_num

        #初始化输入层权值、隐藏层权值
        self.w1 = 0.2 * np.random.random((inp_num, hid_num)) - 0.1
        self.w2 = 0.2 * np.random.random((hid_num, out_num)) - 0.1
        
        # 初始化输入层偏执向量、输出层偏置向量
        self.hid_offset = np.zeros(hid_num)
        self.out_offset = np.zeros(out_num)

    def activation_function(self, x):
        """激活函数 sigmoid"""
        return expit(x)
    
    def normalization(self, x):
        """数据进行归一化处理"""
        return (x/255 * 0.99 + 0.01)
    
    def get_err(self, e):
        """计算总的误差"""
        return 0.5 * np.dot(e, e)

    def train_network(self, iteration, inp_learn_rate, hid_learn_rate, label_data, gui):
        """训练神经网络"""

        ### 记录训练进度
        pro_len = 0
        # 创建训练进度标题并且设置样式
        progress_label = tk.Label(gui.window_name, text='训练进度:', font=('微软雅黑', 15)).grid(row=51, column=1)
        # 创建训练进度的进度条控件
        p1=ttk.Progressbar(gui.window_name, length=200,cursor='spider',mode="determinate",orient=tk.HORIZONTAL)
        # 设置进度条控件的位置
        p1.grid(row=51, column=3)
        

        ### 训练神经网络

        # 将数据的标签与数据分离
        labels = label_data[:,0]
        data = label_data[:,1:]

        # 对数据进行归一化 处理
        normal_data = self.normalization(data)

        # 样本总数
        data_numbers = len(normal_data)

        # 记录总的误差和
        total_errors = []

        # 开始训练
        for j in range(iteration):
            for count in range(data_numbers):

                
                # 对期望输出值进行处理 
                t_label = np.zeros(self.out_num) + 0.01
                t_label[labels[count]] = 0.99


                ## 神经元向前传播

                # 计算隐藏层的值,并计算其激活值
                hide_value = np.dot(normal_data[count], self.w1) + self.hid_offset 
                hid_act = self.activation_function(hide_value)

                # 计算输出层值,并接受其激活值
                out_value = np.dot(hid_act, self.w2)
                out_act = self.activation_function(out_value)


                ## 反向传播,调整参数

                # 输出值与真值之间的误差
                e = t_label - out_act
                out_delta = e * out_act * (1- out_act)
                hid_delta = hid_act * (1 - hid_act) * np.dot(self.w2, out_delta)

                # 记录总误差
                if (j == 0)and( count % 50 == 0 ) and ((count < 2000 and count > 1000 and self.get_err(e) <= 0.3) or (count >= 2000 and count < 5000 and self.get_err(e) <=0.2) or (count >= 5000 and count <= 10000 and self.get_err(e)<0.08) or (count > 10000 and self.get_err(e) < 0.02 )):
                    total_errors.append(self.get_err(e))
                
                # 更新隐藏层到输出层之间的权向量
                for i in range(0, self.out_num):
                    self.w2[:,i] += hid_learn_rate * out_delta[i] * hid_act
                
                # 更新输入层到隐含层之间的权向量
                for i in range(0, self.hid_num):
                    self.w1[:,i] += inp_learn_rate * hid_delta[i] * normal_data[count]

                # 更新偏置值
                self.out_offset += hid_learn_rate * out_delta
                self.hid_offset += inp_learn_rate * hid_delta
                
                #更新进度条
                if count % 300 == 0:
                    p1["value"]=pro_len+0.5
                    pro_len=pro_len+0.5
                    gui.window_name.update()
        
        #绘制误差曲线
        plot_errors(total_errors)
        
        #训练完成提示消息框
        tkmegbox.askquestion('提示','训练已完成,是否开始测试?') 
        

    def read_data(self, network, gui):
        """读取测试数据"""
        
        # 进行测试数据的读取
        test_data = pd.read_csv("mnist_test.csv", header = None)
        test_np = test_data.values
        
        # 测试数据的测试
        self.test_network(test_np, gui) 
        
        # 绘制前20幅图像
        plot_number(test_np, network)


    def test_network(self, label_data, gui):
        """测试神经网络"""

        ### 记录测试训练进度
        pro_len = 0
        # 创建训练进度的标题Label并且设置样式
        progress_label=tk.Label(gui.window_name, text='训练进度:', font=('微软雅黑',15)).grid(row=52, column=1)
        # 创建训练进度的进度条控件并且设置样式
        p1=ttk.Progressbar(gui.window_name,  length=200, cursor='spider',mode="determinate",orient=tk.HORIZONTAL)
        # 设置进度条控件的位置
        p1.grid(row=52, column=3)


        ### 测试神经网络
        # 将数据的标签与数据分离
        labels = label_data[:,0]
        data = label_data[:,1:]

        # 对数据进行归一化 处理
        normal_data = self.normalization(data)

        # 样本总数、识别正确数字总数
        data_numbers = len(data)
        right_numbers = 0

        # 测试数据
        for count in range(data_numbers):

            # 得到识别结果
            output = self.classification_output(normal_data[count])

            #记录测试数据结果 
            if output == labels[count]:
                right_numbers += 1
            if count % 50 == 0:
                    p1["value"]=pro_len+0.5
                    pro_len=pro_len+0.5
                    gui.window_name.update()
        
        
        # 识别正确率
        correct_num = float(right_numbers / data_numbers)
        # 创建正确率的Label并且设置样式
        correct_rate_label = tk.Label(gui.window_name, text="正确率:", font=('微软雅黑',15)).grid(row=53, column=1)
        # 创建显示正确率的Label并且设置样式
        correct_rate_show_label = tk.Label(gui.window_name, text=str(correct_num*100)+"%", font=('微软雅黑', 15)).grid(row=53, column=3)

    
    def show_result(self, network):
        """显示误差曲线和图像分类结果"""

        # 误差曲线图片的路径
        error_img_path='errors_curve.png'
        
        # 测试结果图片的路径
        test_result_img_path='test.png'
        
        # 读取误差曲线图片
        error_img = cv2.imread(error_img_path, cv2.IMREAD_COLOR)
        
        # 读取测试结果图片
        test_result_img=cv2.imread(test_result_img_path, cv2.IMREAD_COLOR)
        
        # 显示误差曲线图片以及测试结果图片
        fig = plt.figure(figsize = (30, 30))
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(error_img, cmap = 'PRGn')
        ax1.axis("off")
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(test_result_img, cmap = 'PRGn')
        ax2.axis("off")
        fig.show()


    def classification_output(self, x):
        """输入一个数字,得到其识别结果"""

        #向神经网络输入图像数据, 得到输出值 
        hid_value = np.dot(x, self.w1) + self.hid_offset
        hid_act = self.activation_function(hid_value)
        out_value = np.dot(hid_act, self.w2) + self.out_offset
        out_act = self.activation_function(out_value)

        # 返回输出结果
        return np.argmax(out_act)