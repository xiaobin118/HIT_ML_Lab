# 机器学习基础讲义（配套实验三）

这份讲义从零开始补机器学习的基础概念，为实验三（k-means 与混合高斯模型）铺路。起点不假定你学过机器学习，实验一/二里出现过的概念都会重新讲清楚。

## 学习路线

| 讲 | 内容 | 状态 |
|---|---|---|
| 01 | 机器学习到底在学什么：三要素、监督/无监督、距离、量纲 | 已完成 |
| 02 | k-means（上）：目标函数、两步迭代、收敛证明、局部最优 | 已完成 |
| 03 | k-means（下）：初始化与 k-means++、怎么选 K、怎么评价聚类 | 已完成 |
| 04 | 从概率看聚类：密度与似然、为什么用高斯、协方差椭圆、数值陷阱 | 已完成 |
| 05 | 混合高斯模型：隐变量、责任度、k-means 是它的极限情形 | 已完成 |
| 06 | EM 算法：Jensen 不等式、E/M 步、单调性证明、分量的移动 | 已完成 |
| 07 | 真实数据的陷阱与完整流程：量纲/特征/维度/不均衡、数值稳定、选 K、评价、八步流程 | 已完成 |

讲义 01 是概念铺垫，02-03 讲 k-means（对应实验要求的第 1 问），04-06 讲 GMM 与 EM（对应第 2 问），07 讲真实数据应用与完整流程（对应第 3 问）。七讲已全部完成。

## 文件说明

```
learn/
├── README.md                          本文件
├── 01_机器学习到底在学什么.md
├── 02_kmeans上_它在最小化什么.md
├── 03_kmeans下_初始化选K与评价.md
├── 04_从概率看聚类.md
├── 05_混合高斯模型与责任度.md
├── 06_EM算法.md
├── 07_真实数据的陷阱与完整流程.md
├── make_fig01.py                      生成讲义 01 的插图
├── make_fig02.py                      生成讲义 02 的插图
├── make_fig03.py                      生成讲义 03 的插图
├── make_fig04.py                      生成讲义 04 的插图
├── make_fig05.py                      生成讲义 05 的插图
├── make_fig06.py                      生成讲义 06 的插图（内含一个最小 EM 实现）
├── make_fig07.py                      生成讲义 07 的插图（内含一个最小 GMM-EM 实现）
├── demo02_kmeans_steps.py             讲义 02 的交互演示（可指定初始中心）
├── review_practice01.md               practice01.py 的代码 review
├── review_kmeans.md                   kmeans.py 的代码 review
├── review_practice03.md               practice03.py（k-means++）的代码 review
├── answer03.md                        讲义 03 六个练习的完整答案（已填）
├── practice01.py                      讲义 01 的练习（待按 review 修改）
├── practice03.py                      讲义 03 的六个练习（可自动跑的那部分已写好）
├── practice04.py                      讲义 04 的五个练习（只有练习 1 需要你写代码）
├── practice05.py                      讲义 05 的五个练习（只有练习 1 需要你写代码）
└── figures/
    ├── fig01_1d_petal_length.png      一维花瓣长度分布（空白区与重叠区）
    ├── fig02_2d_clusters.png          二维三个簇
    ├── fig03_scale_matters.png        量纲问题：各维对距离的贡献占比
    ├── fig04_euclidean.png            欧氏距离的几何含义
    ├── fig05_kmeans_steps.png         手动初始中心的两步迭代过程
    ├── fig06_voronoi.png              决策边界 = 垂直平分线
    ├── fig07_local_optima.png         好/坏初始化的 J 下降曲线对比
    ├── fig08_elbow.png                肘部法：J 曲线与降幅拐点
    ├── fig09_internal_vs_external.png 轮廓系数与 ARI 给出不同的 K
    ├── fig10_gaussian_shape.png       一元高斯的形状与 68-95 规则
    ├── fig11_one_vs_three_gaussians.png 一个高斯 vs 三个高斯（Iris 花瓣长度）
    ├── fig12_2d_ellipse.png           二维高斯的等密度椭圆
    ├── fig13_boundary_vs_bayes.png    k-means 分界 vs Bayes 分界 + 责任度曲线
    ├── fig14_mixture.png              混合高斯：分量与权重
    ├── fig15_jensen.png               Jensen 不等式的几何含义
    ├── fig16_likelihood_bound.png     EM：对数似然与下界两条曲线
    ├── fig17_component_movement.png   EM 迭代中两个分量的移动
    ├── fig18_bic_selection.png        四个数据集上的似然与 BIC 曲线
    ├── fig19_singularity.png          奇异性：方差塌陷导致似然发散
    ├── fig20_dimension_curse.png      维度的诅咒：参数按 D² 增长
    ├── practice01_scatter.png         讲义 01 练习 1 的参考输出
    ├── practice03_iris_K2_vs_K3.png   讲义 03 练习 3：K=2 与 K=3 对比
    ├── practice03_K6_diagnosis.png    讲义 03 练习 6：K=6 的诊断图
    ├── practice04_one_vs_three.png    讲义 04 练习 2：一个高斯 vs 三个高斯
    ├── practice04_ellipse.png         讲义 04 练习 4：马氏距离椭圆
    └── practice04_same_J_diff_shape.png 讲义 04 练习 5：同样 J、不同形状
```

代码依赖 `../code/data_gen.py` 里的数据加载函数，运行前先确认能 `import data_gen`：

```bash
cd learn
python make_fig01.py                              # 讲义 01 的插图（同时打印文中引用的数字）
python make_fig02.py                              # 讲义 02 的插图
python make_fig03.py                              # 讲义 03 的插图（较慢，约 2 分钟）
python make_fig04.py                              # 讲义 04 的插图
python make_fig05.py                              # 讲义 05 的插图
python make_fig06.py                              # 讲义 06 的插图
python make_fig07.py                              # 讲义 07 的插图（较慢，约 2 分钟）
python demo02_kmeans_steps.py                     # 用默认初始中心走两步
python demo02_kmeans_steps.py "0,0" "1,1" "-1,0"  # 自己指定初始中心
python practice03.py 1                            # 讲义 03 练习 1（练习 2 需要你先实现 k-means++）
python practice03.py all                          # 全部六个练习（需要 3~5 分钟）
python practice04.py all                          # 讲义 04 的练习（几秒，练习 1 需先实现）
python practice05.py all                          # 讲义 05 的练习（几秒，练习 1 需先实现）
```

`demo02_kmeans_steps.py` 不是 k-means 的实现（那是你的任务），而是一个**单步执行器**：只做一次分配、一次更新，把过程画出来，让你看清每个点是怎么被重新归属的、中心是怎么移动的。

## 和实验代码的关系

`../docs/` 下那两份文档是**实验报告导向**的（原理推导 + 实现任务书），假设读者已经会 k-means 和 EM。这份讲义不同，它是**学习导向**的：先把概念讲明白，再动代码。等讲义 02-06 学完，再回头读 `../docs/01_原理_kmeans与GMM的EM推导.md` 会顺畅得多；`../docs/02_实现任务与编码路线.md` 里的任务，也应当在那之后再动手。

`../code/` 下的 `kmeans.py` 与 `gmm.py` 目前是空壳（函数体是 `TODO`），这是故意的：等你理解了算法再写。
