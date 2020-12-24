import pandas as pd
import numpy as np
from plot import plot_number
from neuron import Neuron
from gui_tkinter import *

def main():
    
    # 读取训练集csv数据, 转化为np数组
    train_data = pd.read_csv("mnist_train.csv", header = None)
    train_np = train_data.values

    ### 训练神经网络
    
    # 输入层结点数, 隐含层结点数, 输出层结点数
    inp_num = len(train_np[0, 1:])
    hid_num = 20
    out_num = 10
    
    # 初始化神经网络
    network = Neuron(inp_num, hid_num, out_num)
    
    # 调用函数进行窗体的初始化
    gui_func(network, train_np)
    

def gui_func(network, train_np):
    """主窗体"""
    
    # 创建窗体
    window_name = Tk()
    # 创建窗体类接收该窗体
    my_Gui = main_Gui(window_name)
    
    # 初始化窗体类
    my_Gui.init_window(network, train_np)
    window_name.mainloop()


if __name__ == '__main__':
    main()