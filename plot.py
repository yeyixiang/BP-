
import matplotlib.pyplot as plt



def plot_number(data, network):
    """将20幅图像在同一窗口绘制 20 个子图"""

    fig = plt.figure(figsize = (10, 10))
    
    for i in range(20):
        # 获取每个数字图像的RGB数据
        number = data[i, 1:].reshape(28, -1)

        # 每个数字图像绘制一个子图
        ax = fig.add_subplot(4, 5, i+1)
        ax.imshow(number, cmap = "Greys")
        normal_data = network.normalization(data[i, 1:])
        ax.set_title("The number is %s" % (str(network.classification_output(normal_data))), fontsize = 12)
        ax.axis("off")
    plt.savefig('test.png')
    
    
    

def plot_errors(errors):
    """绘制误差曲线"""

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  
    plt.figure(figsize = (10, 10))
    plt.plot(errors, '-',  color = 'r', linewidth = 2)
    plt.ylim([0.0,0.8])
    plt.ylabel('总误差')
    plt.xlabel('迭代次数 /每10次')
    plt.title("误差曲线图")
    plt.savefig('errors_curve.png')


