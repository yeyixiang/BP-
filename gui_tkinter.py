from tkinter import *
from neuron import Neuron

class main_Gui():
    def __init__(self, window_name):
        """初始化窗体类"""
        self.window_name = window_name
    def init_window(self,network, train_np):
        """在窗体类内添加控件"""

        # 将该参数与迭代次数输入框的值进行绑定
        self.str1 = StringVar()
        # 将该参数与输入层学习率输入框的值进行绑定
        self.str2 = StringVar()
        # 将该参数与隐藏层学习率输入框的值进行绑定
        self.str3 = StringVar()
        
        # 创建标题
        self.window_name.title('手写体数字识别')
        # 创建窗体的大小
        self.window_name.geometry('400x300+500+200')
        
        # 创建题目的标题的Label并且设置样式
        self.init_Label = Label(self.window_name, text = '手写体数字识别', font = ('微软雅黑', 15)).grid(row = 0, column = 3)
        # 创建输入参数标题的Label并且设置样式
        self.input_Label1 = Label(self.window_name,text='输入参数:',font=('微软雅黑',15)).grid(row=2, column=1)
        
        # 创建迭代次数标题的Label并且设置样式
        self.init_Label1 = Label(self.window_name,text='迭 代 次 数    :',font=('微软雅黑',15),bd=2,relief=GROOVE).grid(row=3, column=1)
        # 创建迭代次数的输入框并且设置样式
        self.iter_entry = Entry(self.window_name, textvariable=self.str1).grid(row=3, column=3)
        
        # 创建输入层学习率的标题并且设置样式
        self.init_Label2 = Label(self.window_name, text='输入层学习率:', font=('微软雅黑',15), bd=2, relief=GROOVE).grid(row=5, column=1)
        # 创建输入层学习率的输入框并且设置样式
        self.inp_len_rate_entry = Entry(self.window_name, textvariable=self.str2).grid(row=5, column=3)
        
        # 创建隐藏层学习率的标题并且设置样式
        self.init_Label3 = Label(self.window_name, text='隐藏层学习率:', font=('微软雅黑',15), bd=2, relief=GROOVE).grid(row=7, column=1)
        # 创建隐藏层学习率的输入框并且设置样式
        self.hid_len_rate_entry = Entry(self.window_name, textvariable=self.str3).grid(row=7, column=3)
        
        # 创建开始训练按钮并且绑定事件函数
        self.start_button = Button(self.window_name, text='开始训练', font=('微软雅黑',10), command=lambda : network.train_network(int(self.str1.get()), float(self.str2.get()), float(self.str3.get()), train_np, self)).grid(row=50, column=1)
        # 创建开始测试按钮并且绑定事件函数
        self.start_button = Button(self.window_name, text='开始测试', font=('微软雅黑',10), command=lambda : network.read_data(network, self)).grid(row=50, column=3)
        # 创建显示结果按钮并且绑定事件函数
        self.start_button = Button(self.window_name, text='显示结果', font=('微软雅黑',10),command=lambda : network.show_result(network)).grid(row=50, column=5)
    
    
