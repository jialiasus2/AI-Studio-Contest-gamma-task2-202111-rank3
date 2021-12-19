# 飞桨常规赛：黄斑中央凹定位（GAMMA挑战赛任务二） - 11月第3名方案
鸣沙山下.伽利略<br>
比赛地址：![https://aistudio.baidu.com/aistudio/competition/detail/120/0/introduction](https://aistudio.baidu.com/aistudio/competition/detail/120/0/introduction)

## 1. 复现说明

### 1.1 安装依赖库
```bash
!pip install paddlepaddle-gpu
!pip install paddleseg opencv-python visualdl
```
在安装飞桨过程中遇到任何问题可以参考![https://www.paddlepaddle.org.cn/](https://www.paddlepaddle.org.cn/)

### 1.2 下载并解压数据集
```bash
# 下载比赛数据，报名比赛后在比赛页面可查看下载地址
!wget https://xxxxxxxxxxxxxxx/task2_Fovea_localization.zip
# 解压至competition_data文件夹
!unzip task2_Fovea_localization.zip -d competition_data
```

### 1.3 训练
**注意**：会覆盖之前训练好的checkpoint
```bash
!cd ~/work && python -W ignore train.py
```

### 1.4 预测

```bash
!cd ~/work && python -W ignore predict.py
```

## 2. 代码说明
### 2.1 建模思路
题目要求检测黄斑中央凹点，参考baseline将其建模为回归问题。<br>
backbone选用飞桨自带的Resnet50，输出为归一化的xy坐标值。<br>

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
1. 全图取中使宽高一致，再缩放为512*512
2. 训练数据增强选用了颜色抖动和随机翻转
3. 学习率采用warmup和线形递减，基准学习率为1e-3
4. 优化器采用Momentum
5. 后处理时在局部邻域取灰度最低点作为最终结果

## 3. 不足与改进
其实原本想借鉴YOLO的思路，把题目建模为“分块回归”问题，即把黄斑中央凹可能存在的区域划分为若干网格，首先通过分类确定中央凹点在哪个网格内，再通过回归确定最终位置，如图所示：<br>
![](https://ai-studio-static-online.cdn.bcebos.com/a62e08d64f0c445cb03784d372737cfe4984162efcaa49c5b20df403468a6050)<br>
但实现以后发现效果不如直接回归的，大概是参数调的不够好吧。<br>
另外，后处理假设黄斑中央凹点为局部邻域灰度最低点，这个假设也有点小问题，想要进一步提高分数可能要放弃该假设。


