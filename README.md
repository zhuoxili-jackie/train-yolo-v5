# train-yolo-v5 — 人群异常聚集检测

> 第一届浙江省大学生人工智能竞赛参赛项目

基于 **YOLOv5** 的行人检测 + **DBSCAN** 密度聚类，用于在图像中**检测人群并圈出异常聚集区域**。

---

## 1. 这是干啥的（What）

整体流程分两步：

1. **检测**：用 YOLOv5s 在图像里检测所有「人」（只有一个类别 `person`），输出每个人的中心点。
2. **聚类判异常**：对所有人的中心点做 DBSCAN 聚类，若某个簇内的人数 ≥ 阈值，就判定为「异常聚集」，在结果图上用**黄色圆圈**框出来。
   - 普通点：绿色
   - 属于异常簇的点：红色
   - 异常簇：黄色外接圆（半径放大 5%）

可调参数（在 `inference_scripts/predict.py` 顶部）：

| 参数                    | 含义                       | 默认值 |
| ----------------------- | -------------------------- | ------ |
| `R`                     | DBSCAN 邻域半径（像素）    | 80     |
| `MinPts`                | 成为核心点所需的最少邻居数 | 3      |
| `ClusterSize_threshold` | 判定为异常簇的最小人数     | 5      |

---

## 2. 怎么用（How）

### 环境准备

```bash
# 1. 安装本项目脚本依赖
pip install -r inference_scripts/requirements.txt

# 2. 安装 YOLOv5 自身依赖
pip install -r yolov5/requirements.txt
```

> 依赖：numpy、opencv-python、matplotlib、scikit-learn、torch（以及 yolov5 的依赖）。

### 训练模型

```bash
cd inference_scripts
python train.py
```

`train.py` 实际是对 `yolov5/train.py` 的封装，固定参数为：
`--img 640 --batch 16 --epochs 100`，基础权重 `yolov5/yolov5s.pt`，
数据集配置 `data.yaml`，结果保存到 `runs/crowd_yolov5s/`。
训练得到的最佳权重在 `runs/crowd_yolov5s/weights/best.pt`。

### 推理 + 异常检测

```bash
cd inference_scripts
python predict.py
```

`predict.py` 会：

1. 调 `yolov5/detect.py` 对 `TEST_DIR` 里的图片做检测，导出标签到 `runs/crowd_yolov5s_detect/labels/`；
2. 读取检测框中心点，做 DBSCAN 聚类，圈出异常聚集；
3. 把可视化结果写到 `RESULT_SAVE_DIR`（如 `image_results/20251016_v1/`）。

> ⚠️ 运行前请按需修改 `predict.py` 顶部的 `TEST_DIR`、`RESULT_SAVE_DIR`、`TRAINED_WEIGHT` 路径，
> 并确保输出目录已存在（脚本不会自动创建）。

---

## 3. 目录结构

```
train-yolo-v5/
├── data.yaml                 # 数据集配置：train/val 路径，nc=1，names=['person']
├── datasets/                 # 训练数据（YOLO 格式）
│   ├── images/{train,val}/   #   图片：1550 训练 / 223 验证
│   └── labels/{train,val}/   #   标签 txt（每行：cls cx cy w h，归一化）
├── inference_scripts/        # 本项目自己的脚本
│   ├── train.py              #   训练入口（封装 yolov5/train.py）
│   ├── predict.py            #   推理 + DBSCAN 聚类 + 可视化
│   ├── utils.py              #   （当前为空）
│   ├── playground.ipynb      #   调试用 notebook
│   └── requirements.txt
├── model/
│   ├── configs/              # 模型结构配置（yolov5s 架构）
│   └── weights/              # 已训练权重 crowd_detector.pt（约 89MB）
├── runs/                     # YOLOv5 训练/检测输出
├── image_results/            # 异常检测可视化结果（按日期版本命名）
└── yolov5/                   # 内置的 Ultralytics YOLOv5 源码
```

---

## 4. 数据格式

- 配置文件 `data.yaml`：`nc: 1`，`names: ['person']`
- 标签为标准 YOLO 格式，每行 `class cx cy w h`（坐标已经归一化到 0~1）
- 图片与标签按文件名一一对应（`xxx.jpg` ↔ `xxx.txt`）

---

## 致谢

行人检测基于 [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)（AGPL-3.0）。
