# 飞桨常规赛：黄斑中央凹定位（GAMMA挑战赛任务二） - 11月第3名方案
鸣沙山下.伽利略<br>
比赛地址：[https://aistudio.baidu.com/aistudio/competition/detail/120/0/introduction](https://aistudio.baidu.com/aistudio/competition/detail/120/0/introduction)

## 1. 赛题分析
GAMMA挑战赛是由百度在MICCAI2021研讨会OMIA8上举办的国际眼科赛事。MICCAI是由国际医学图像计算和计算机辅助干预协会 (Medical Image Computing and Computer Assisted Intervention Society) 举办的跨医学影像计算和计算机辅助介入两个领域的综合性学术会议，是该领域的顶级会议。OMIA是百度在MICCAI会议上组织的眼科医学影像分析 (Ophthalmic Medical Image Analysis) 研讨会，至今已举办八届。<br>
### 1.1 题目
对2D眼底图像中黄斑中央凹进行定位。<br>
本任务的目的是预测黄斑中央凹在图像中的坐标值。若图像中黄斑中央凹不可见，坐标值设为(0, 0)，否则需预测出中央凹在图像中的坐标值。<br>
![](https://ai.bdstatic.com/file/712D43A2022644428756653A194AAB64)
### 1.2 数据分析
数据集由中国广州中山大学中山眼科中心提供，数据集中包含200个2D眼底彩照样本，分别为：训练集100个，测试集100个。<br>
样本数虽然不多，但测试集和训练集数据一致性较好，实际训练时没有发现过拟合现象。图像分辨率大部分为2992 * 2000，少量为1956 * 1934，图片分辨率高。<br>
中央凹标注单位为像素，精度精确到小数点后5-6位（确实不知道是否真的需要这么高的精度嘛？）。虽然说明中有说中央凹不可见时坐标值设为(0, 0)，但标注中并未发现这种情况，暂时只能认为测试集中也不存在这种情况。

## 2. 方案说明
### 2.1 建模思路
题目要求检测黄斑中央凹点，参考baseline将其建模为回归问题。<br>
backbone选用paddleseg自带的Resnet50vd，输出为归一化的xy坐标值。<br>
考虑到中央凹位置坐标范围有限，这里将坐标映射到[0.3, 0.7]的区间范围内再归一化到[0, 1]区间。

### 2.2 代码结构
| 代码 | 功能 |
|-|-|
| config.py | 参数设置 |
| utils.py | 功能函数 |
| my_dataset.py | 数据集 |
| my_model.py | 模型 |
| train.py | 训练主程序 |
| predict.py | 预测主程序 |

### 2.3 一些细节
1. 预处理时全图取中使宽高一致，再缩放为512*512
2. 训练数据增强选用了颜色抖动和随机翻转
3. 学习率采用warmup和线形递减，基准学习率为1e-3
4. 优化器采用Momentum
5. 后处理时在局部邻域取灰度最低点作为最终结果

### 2.4 可视化
不得不说visualdl可视化太棒了，哪怕只是简单的画一下loss和evaluatie曲线，都感觉对训练过程把控好很多。

## 3. 复现说明

### 3.1 安装依赖库
```bash
!pip install paddlepaddle-gpu
!pip install paddleseg opencv-python visualdl
```
在安装飞桨过程中遇到任何问题可以参考[https://www.paddlepaddle.org.cn/](https://www.paddlepaddle.org.cn/)

### 3.2 下载并解压数据集
```bash
# 下载比赛数据，报名比赛后在比赛页面可查看下载地址
!wget https://xxxxxxxxxxxxxxx/task2_Fovea_localization.zip
# 解压至competition_data文件夹
!unzip task2_Fovea_localization.zip -d competition_data
```

### 3.3 训练
**注意**：会覆盖之前训练好的checkpoint
```bash
!cd ~/work && python -W ignore train.py
```

### 3.4 预测

```bash
!cd ~/work && python -W ignore predict.py
```

## 4. 不足与改进
其实原本想借鉴YOLO的思路，把题目建模为“分块回归”问题，即把黄斑中央凹可能存在的区域划分为若干网格，首先通过分类确定中央凹点在哪个网格内，再通过回归确定最终位置，如图所示：<br>
![](https://ai-studio-static-online.cdn.bcebos.com/a62e08d64f0c445cb03784d372737cfe4984162efcaa49c5b20df403468a6050)<br>
但实现以后发现效果不如直接回归的，大概是参数调的不够好吧。<br>
另外，后处理假设黄斑中央凹点为局部邻域灰度最低点，这个假设也有点小问题，想要进一步提高分数可能要放弃该假设。


