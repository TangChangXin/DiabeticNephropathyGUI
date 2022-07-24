import sys
import torch
from PIL import Image
from torchvision import transforms
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from model import 有监督分类阶段6MR多批量 as 有监督分类模型


class 糖肾诊断系统(QWidget):
    def __init__(self):
        super(糖肾诊断系统, self).__init__()

        self.setGeometry(300, 300, 900, 900) # 设置窗口位置和大小
        self.setWindowTitle("糖肾诊断系统") # 设置窗口标题

        self.垂直盒子 = QVBoxLayout() # 创建一个垂直布局，存放所有窗口部件
        self.选择图像 = QPushButton('选择en-face OCTA图像', self)
        self.垂直盒子.addWidget(self.选择图像, alignment=Qt.AlignCenter)
        self.垂直盒子.addStretch(1)

        self.糖肾概率 = QLabel('糖肾概率：')
        self.垂直盒子.addWidget(self.糖肾概率, alignment=Qt.AlignCenter)
        self.垂直盒子.addStretch(1)
        # self.显示糖肾概率 = QLabel()
        # self.垂直盒子.addWidget(self.显示糖肾概率, alignment=Qt.AlignCenter)
        # self.垂直盒子.addStretch(1)

        self.非糖肾概率 = QLabel('非糖肾概率：')
        self.垂直盒子.addWidget(self.非糖肾概率, alignment=Qt.AlignCenter)
        self.垂直盒子.addStretch(1)
        # self.显示非糖肾概率 = QLabel()
        # self.垂直盒子.addWidget(self.显示非糖肾概率, alignment=Qt.AlignCenter)
        # self.垂直盒子.addStretch(1)

        self.显示OCTA图像 = QLabel()
        self.垂直盒子.addWidget(self.显示OCTA图像, alignment=Qt.AlignCenter)
        self.垂直盒子.addStretch(15)

        self.setLayout(self.垂直盒子)
        self.show()

        self.选择图像.clicked.connect(self.open_en_face_image)

    def open_en_face_image(self):
        文件名, 后缀名 = QFileDialog.getOpenFileName(self, '打开图像文件', './', "图像 (*.png *.jpg *.bmp)")
        if 文件名:
            硬件设备 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # print('使用设备', 硬件设备)

            待诊断OCTA图像 = Image.open(文件名).convert('L') # 转换成单通道灰度图
            # 待诊断OCTA图像
            待诊断OCTA图像 = transforms.ToTensor()(待诊断OCTA图像)
            待诊断OCTA图像 = torch.unsqueeze(待诊断OCTA图像, dim=0) # 调整成批量、通道、高度、宽度的形式
            待诊断OCTA图像 = 待诊断OCTA图像.to(硬件设备)
            分类模型 = 有监督分类模型(2)  # 生成模型，需传入分类数目
            分类模型.to(硬件设备)
            分类模型.load_state_dict(torch.load('Model1.pth', map_location=硬件设备))  # 加载模型参数

            with torch.no_grad():
                输出 = 分类模型(待诊断OCTA图像)
                输出 = torch.squeeze(输出)
                预测概率 = torch.softmax(输出, dim=0)
                # print(预测概率)

            self.显示OCTA图像.setPixmap(QPixmap(文件名))
            self.糖肾概率.setText('糖肾概率：' + str(预测概率[1].item()))
            self.非糖肾概率.setText('非糖肾概率：' + str(预测概率[0].item()))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    窗口 = 糖肾诊断系统()
    app.exec()





