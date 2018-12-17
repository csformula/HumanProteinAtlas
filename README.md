# Human Protein Atlas Image Classification
Kaggle图像分类比赛，数据集由医疗机构提供，旨在识别不同蛋白质种类

- 数据分布高度不平衡，自定义focal loss
- 输入为四通道，非传统三通道图片
- 使用了大量数据分析功能
- 多目标分类，迁移学习，训练采用k折交叉验证
- 使用macro F1 score作为精度标准
- 存取ckeckpoint文件
