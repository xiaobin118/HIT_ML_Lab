"""数据生成与加载工具。

本模块提供三类东西：
1. 合成数据生成（任务 1.1 为脚手架，函数体已给出）；
2. 多元高斯对数密度（任务 1.2 需要你实现，函数体留空）；
3. UCI 数据集加载（脚手架，函数体已给出）。

接口约定：所有密度类函数返回自然对数，函数名以 `_logpdf` 结尾。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 合成数据

#: 三个真实分量的均值。刻意选成互相分离，便于观察 EM 能否恢复参数。
TRUE_MEANS = np.array(
    [
        [-4.0, -3.0],
        [0.0, 4.5],
        [5.0, -2.0],
    ]
)

#: 三个真实分量的协方差：各向同性圆、横躺的细长椭圆、45 度倾斜的椭圆。
TRUE_COVS = np.array(
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[2.5, 0.0], [0.0, 0.25]],
        [[1.2, 0.9], [0.9, 1.2]],
    ]
)

TRUE_WEIGHTS = np.array([0.4, 0.35, 0.25])


def _regularize(cov, reg=1e-6):
    """给协方差对角线加一个与尺度相关的小量，保证正定。

    对每个矩阵单独处理：加 reg * trace/D 而不是固定常数，这样量纲不同的
    数据集都能得到合适的扰动强度。D 为矩阵阶数。
    """
    cov = np.asarray(cov, dtype=np.float64)
    d = cov.shape[-1]
    scale = reg * np.trace(cov, axis1=-2, axis2=-1) / d
    return cov + scale[..., None, None] * np.eye(d) if cov.ndim > 2 else cov + scale * np.eye(d)


def make_synthetic_data(means, covs, weights=None, n_samples=1500, seed=0):
    """按混合高斯模型生成数据。

    Parameters
    ----------
    means : (K, D) array
        各分量的均值。
    covs : (K, D, D) array
        各分量的协方差。
    weights : (K,) array or None
        混合系数，None 表示等权。
    n_samples : int
        样本总数。
    seed : int
        随机种子。

    Returns
    -------
    X : (n_samples, D) float64
    y : (n_samples,) int64
        每个样本的真实分量编号，取值 0..K-1。
    """
    means = np.asarray(means, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)
    k = means.shape[0]
    if weights is None:
        weights = np.full(k, 1.0 / k)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    rng = np.random.default_rng(seed)
    # 第一步：为每个样本抽一个分量指示 z_n。
    z = rng.choice(k, size=n_samples, p=weights)

    # 第二步：按分量分组，一次性抽完该分量的全部样本。
    # 逐样本循环也能写对，但按分量批量抽取可以利用 NumPy 的向量化。
    x = np.empty((n_samples, means.shape[1]), dtype=np.float64)
    for j in range(k):
        idx = np.flatnonzero(z == j)
        if idx.size == 0:
            continue
        x[idx] = rng.multivariate_normal(means[j], covs[j], size=idx.size)

    return x, z.astype(np.int64)


def normal_logpdf(x, mean, cov, reg=1e-6):
    """多元高斯分布的对数密度。

    Parameters
    ----------
    x : (N, D) array
        待求值的点，允许是单点 (D,) 形式。
    mean : (D,) array
    cov : (D, D) array
    reg : float
        正则化系数，见 `_regularize`。

    Returns
    -------
    (N,) array
        log N(x_n | mean, cov)。当输入是单点 (D,) 时返回标量。

    实现提示
    --------
    1. 用 `_regularize(cov, reg)` 处理协方差，再做 Cholesky 分解 L。
    2. log|Sigma| = 2 * sum(log(diag(L)))，不要用 np.linalg.det。
    3. 马氏距离平方用三角方程组求解：令 diff = x - mean，
       solve(L, diff.T) 得到 (D, N)，取列平方和。
    4. 完整公式（原理文档 3.1 节）：
       log N(x|mu,Sigma) = -D/2*log(2*pi) - 1/2*log|Sigma|
                           - 1/2*(x-mu)^T Sigma^{-1} (x-mu)
    """
    raise NotImplementedError("任务 1.2：请实现 normal_logpdf")


# UCI 数据集加载


def load_iris(standardize=False, feature_idx=None):
    """加载 UCI Iris 数据集。

    Parameters
    ----------
    standardize : bool
        是否按列做 z-score 标准化。Iris 四列量纲相近，默认不标准化。
    feature_idx : list[int] or None
        只保留指定列。例如 [2, 3] 表示只用花瓣长度与花瓣宽度，
        用于画二维协方差椭圆。

    Returns
    -------
    X : (150, D) float64
    y : (150,) int64
        类别编号 0/1/2（原始顺序 setosa, versicolor, virginica）。
    feature_names : list[str]
    """
    path = DATA_DIR / "iris.data"
    rows, labels = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:  # 文件末尾有空行
                continue
            parts = line.split(",")
            rows.append([float(v) for v in parts[:4]])
            labels.append(parts[4])
    x = np.asarray(rows, dtype=np.float64)

    classes = sorted(set(labels))
    y = np.asarray([classes.index(v) for v in labels], dtype=np.int64)

    names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    if feature_idx is not None:
        x = x[:, feature_idx]
        names = [names[i] for i in feature_idx]

    if standardize:
        x = _standardize(x)

    return x, y, names


def load_seeds(standardize=True, feature_idx=None):
    """加载 UCI Seeds 数据集（七维，三类小麦种子）。

    原始文件用制表符分隔，最后一列是类别 1/2/3。七列量纲差别很大
    （面积约 10-21，紧致度约 0.8-0.92），因此默认做 z-score 标准化，
    否则欧氏距离会被面积一列主导。

    Returns
    -------
    X : (210, D) float64
    y : (210,) int64
        类别编号 0/1/2。
    feature_names : list[str]
    """
    path = DATA_DIR / "seeds.data"
    raw = np.loadtxt(path, dtype=np.float64, delimiter=None)
    x = raw[:, :7]
    y = raw[:, 7].astype(np.int64) - 1  # 原始标签是 1/2/3

    names = [
        "area",
        "perimeter",
        "compactness",
        "kernel_length",
        "kernel_width",
        "asymmetry",
        "groove",
    ]
    if feature_idx is not None:
        x = x[:, feature_idx]
        names = [names[i] for i in feature_idx]

    if standardize:
        x = _standardize(x)

    return x, y, names


def _standardize(x):
    """按列做 z-score 标准化，对零方差列做保护。

    常数列（方差为 0）会让除法产生 nan 并污染整个下游计算。真实数据集里
    这种情况很常见：optdigits 的 8x8 图像中有若干像素位置恒为 0（笔迹从不
    触及那些边界像素），维数越高越容易碰到。处理方式是把标准差为 0 的列
    直接置 0（该列本就不含任何区分信息），而不是让它变成 nan。
    """
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    safe = sd > 1e-12
    out = np.zeros_like(x)
    out[:, safe] = (x[:, safe] - mu[safe]) / sd[safe]
    return out


def save_csv(path, header, array):
    """把二维数组写成 CSV，供报告和绘图使用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(array), delimiter=",", header=",".join(header), comments="")
    return path


# ARFF 加载（OpenML 下载的数据集用这个格式）


def load_arff(path, standardize=False, feature_idx=None, max_samples=None, seed=0):
    """读取稠密 ARFF 文件，最后一个属性作为类别标签。

    只支持稠密格式（我们下载的这几个都是），稀疏 ARFF 会直接报错而不是
    静默给出错误结果。数值属性按 float 解析，标称属性（类别列）按出现
    顺序映射成 0..C-1 的整数。

    类别列的位置不能假定在最后。OpenML 上多数数据集的 `class` 是最后一个
    属性，但 wine 把它放在第一个，而 electricity 还有一个名为 `day` 的标称
    属性混在中间。因此这里显式定位类别列：优先取名为 class 的标称属性，
    否则取第一个标称属性，其余属性一律要求能解析成数值。

    Parameters
    ----------
    max_samples : int or None
        只随机抽取这么多样本。样本量很大的数据集（如 letter 的 20000 条）
        在纯 NumPy 实现下跑 EM 会很慢，抽样可以在保留结构的前提下缩短
        运行时间；设成 None 则全量使用。

    Returns
    -------
    X : (N, D) float64
    y : (N,) int64
    feature_names : list[str]
    class_names : list[str]
    """
    path = Path(path)
    attr_names, attr_types = [], []
    rows, labels, class_names = [], [], []
    class_index = {}
    in_data = False
    class_idx = None

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@data"):
                in_data = True
                continue
            if not in_data:
                if low.startswith("@attribute"):
                    # @attribute 'name' {a,b,c}  或  @attribute name numeric
                    body = line[len("@attribute"):].strip()
                    if body[0] in "'\"":
                        quote = body[0]
                        end = body.index(quote, 1)
                        name = body[1:end]
                        rest = body[end + 1:].strip()
                    else:
                        parts = body.split(None, 1)
                        name = parts[0]
                        rest = parts[1].strip() if len(parts) > 1 else ""
                    attr_names.append(name)
                    attr_types.append(rest)
                continue

            if line.startswith("{"):
                raise ValueError(f"{path.name} 是稀疏 ARFF 格式，本加载器不支持")

            if class_idx is None:
                class_idx = _find_class_column(attr_names, attr_types)
                feature_cols = [i for i in range(len(attr_names)) if i != class_idx]

            values = _split_arff_row(line)
            if len(values) != len(attr_names):
                continue  # 跳过长度异常的行
            label = values[class_idx].strip().strip("'\"")
            if label not in class_index:
                class_index[label] = len(class_names)
                class_names.append(label)
            labels.append(class_index[label])
            rows.append([_to_float(values[i]) for i in feature_cols])

    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    names = [attr_names[i] for i in feature_cols]

    if max_samples is not None and len(x) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=max_samples, replace=False)
        x, y = x[idx], y[idx]

    if feature_idx is not None:
        x = x[:, feature_idx]
        names = [names[i] for i in feature_idx]

    if standardize:
        x = _standardize(x)

    return x, y, names, class_names


def _split_arff_row(line):
    """按逗号切分 ARFF 数据行，尊重单引号/双引号包裹的字段。"""
    out, buf, quote = [], [], None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
            else:
                buf.append(ch)
        elif ch in "'\"":
            quote = ch
        elif ch == ",":
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out


def _find_class_column(attr_names, attr_types):
    """定位类别列的下标。

    判断顺序：名为 class（忽略大小写与引号）的标称属性 -> 第一个标称属性。
    标称属性的标志是类型串以 `{` 开头（枚举声明）。找不到就报错，而不是
    退回"取最后一列"这种在 wine 上会静默出错的行为。
    """
    nominal = [i for i, t in enumerate(attr_types) if t.strip().startswith("{")]
    for i in nominal:
        if attr_names[i].strip().strip("'\"").lower() == "class":
            return i
    if nominal:
        return nominal[0]
    raise ValueError(f"在属性 {attr_names} 中找不到标称（类别）属性")


def _to_float(v):
    """把 ARFF 字段转成浮点数，把缺失值标记按 0 处理并允许整数字段。"""
    s = v.strip().strip("'\"")
    if s == "" or s == "?":
        return 0.0
    return float(s)


# ---------------------------------------------------------------------------
# 更大规模的数据集加载器


def load_optdigits(standardize=True, feature_idx=None, max_samples=None, seed=0):
    """手写数字光学识别（Optical Recognition of Handwritten Digits）。

    5620 个样本（训练集），64 维（8x8 像素灰度，取值 0-16），10 个类别。
    这是"更大样本量"的代表：1500 -> 5620，聚类结构仍清晰，但簇数从 3 变成
    10，k-means 的重启次数和 EM 的迭代轮数都会明显上升。
    """
    return load_arff(
        DATA_DIR / "downloads" / "optdigits.arff",
        standardize=standardize,
        feature_idx=feature_idx,
        max_samples=max_samples,
        seed=seed,
    )


def load_pendigits(standardize=True, feature_idx=None, max_samples=None, seed=0):
    """手写数字笔迹识别（Pen-Based Recognition of Handwritten Digits）。

    10992 个样本，16 维（8 个轨迹点的 x/y 坐标），10 个类别。样本量比
    optdigits 大一倍，维度却低得多，因此单轮 EM 更快，适合观察迭代轮数
    随样本量的变化。
    """
    return load_arff(
        DATA_DIR / "downloads" / "pendigits.arff",
        standardize=standardize,
        feature_idx=feature_idx,
        max_samples=max_samples,
        seed=seed,
    )


def load_satimage(standardize=True, feature_idx=None, max_samples=None, seed=0):
    """Landsat 卫星图像（Statlog Satimage）。

    6430 个样本，36 维（3x3 邻域 4 个光谱波段的像素值），6 个类别。
    类别分布不均衡，能观察 GMM 的混合系数是否反映出这种不均衡。
    """
    return load_arff(
        DATA_DIR / "downloads" / "satimage.arff",
        standardize=standardize,
        feature_idx=feature_idx,
        max_samples=max_samples,
        seed=seed,
    )


def _load_segmentation_file(path, n_features=19):
    """读取图像分割数据集的原始文件（类别在第一列）。

    这个文件的排版不能靠 `skip_header` 处理：前 3 行是 `;;;` 注释，第 4 行是
    19 个属性名（注意：属性名行只有 19 列，而数据行有 20 列，类别在第一列），
    第 5 行是空行，数据从第 6 行开始。因此这里逐行判断：跳过注释与空行，凡是
    字段数等于 n_features + 1 的行才当作数据。
    """
    labels, rows = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != n_features + 1:
                continue  # 属性名行（19 列）会在这里被跳过
            labels.append(parts[0])
            rows.append([float(v) for v in parts[1:]])

    x = np.asarray(rows, dtype=np.float64)
    classes = sorted(set(labels))
    y = np.asarray([classes.index(v) for v in labels], dtype=np.int64)
    return x, y, classes


def load_segmentation(standardize=True, feature_idx=None, part="train", seed=0):
    """图像分割数据集（Image Segmentation）。

    训练集 210 个样本、测试集 2100 个样本，19 维（图像的区域中心、边缘密度、
    颜色均值等特征），7 个类别（砖墙、水泥、树叶、草地、小路、天空、窗户）。
    训练集和测试集分开存放，因此可以做一个在实验一/二里做过的动作：在训练集
    上估计参数，在测试集上评估。对聚类来说这是少见的"训练/测试分离"案例——
    用训练集学到的簇中心划分测试集样本，可以说明聚类是否只是在记忆训练数据
    的分布。
    """
    path = DATA_DIR / ("segmentation.data" if part == "train" else "segmentation.test")
    x, y, _ = _load_segmentation_file(path)
    names = [f"attr_{i}" for i in range(x.shape[1])]

    if feature_idx is not None:
        x = x[:, feature_idx]
        names = [names[i] for i in feature_idx]

    if standardize:
        x = _standardize(x)

    return x, y, names


def load_wdbc(standardize=True, feature_idx=None):
    """威斯康星乳腺癌诊断（Breast Cancer Wisconsin Diagnostic）。

    569 个样本，30 维（细胞核的半径/纹理/周长等 10 个特征的均值、标准差、
    最差值），2 个类别（恶性 212 / 良性 357）。只有两类，适合验证"K 设成
    真实类别数"之外的情形：K=2 时 GMM 与 k-means 的差别最小，因为两个
    类别的形状差异有限。
    """
    path = DATA_DIR / "wdbc.data"
    raw = np.genfromtxt(path, delimiter=",", dtype=str)
    # 第 0 列是 ID，第 1 列是类别 M/B，第 2 列起是 30 个特征
    labels = raw[:, 1]
    x = raw[:, 2:].astype(np.float64)
    classes = sorted(set(labels.tolist()))
    y = np.asarray([classes.index(v) for v in labels], dtype=np.int64)
    names = [f"feat_{i}" for i in range(x.shape[1])]

    if feature_idx is not None:
        x = x[:, feature_idx]
        names = [names[i] for i in feature_idx]

    if standardize:
        x = _standardize(x)

    return x, y, names


#: 数据集注册表：名称 -> (加载函数, 期望样本数, 期望特征数, 类别数, 说明)
DATASETS = {
    "iris_2d": (lambda **kw: load_iris(feature_idx=[2, 3], **kw), 150, 2, 3, "Iris 花瓣两维，可画椭圆"),
    "iris_4d": (lambda **kw: load_iris(**kw), 150, 4, 3, "Iris 全部四维"),
    "seeds_7d": (lambda **kw: load_seeds(**kw), 210, 7, 3, "Seeds 七维，需标准化"),
    "seg_train": (lambda **kw: load_segmentation(part="train", **kw), 210, 19, 7, "图像分割训练集"),
    "seg_test": (lambda **kw: load_segmentation(part="test", **kw), 2100, 19, 7, "图像分割测试集"),
    "wdbc": (lambda **kw: load_wdbc(**kw), 569, 30, 2, "乳腺癌诊断，二分类"),
    "wine": (lambda **kw: load_arff(DATA_DIR / "downloads" / "wine.arff", **kw), 178, 13, 3, "葡萄酒，类别列在首位"),
    "vehicle": (lambda **kw: load_arff(DATA_DIR / "downloads" / "vehicle.arff", **kw), 846, 18, 4, "车辆轮廓，四分类"),
    "optdigits": (lambda **kw: load_optdigits(**kw), 5620, 64, 10, "手写数字，64 维 10 类"),
    "satimage": (lambda **kw: load_satimage(**kw), 6430, 36, 6, "卫星图像，类别不均衡"),
    "pendigits": (lambda **kw: load_pendigits(**kw), 10992, 16, 10, "笔迹数字，样本量最大"),
    "letter": (lambda **kw: load_arff(DATA_DIR / "downloads" / "letter.arff", **kw), 20000, 16, 26, "英文字母，26 类"),
    "diabetes": (lambda **kw: load_arff(DATA_DIR / "downloads" / "diabetes.arff", **kw), 768, 8, 2, "糖尿病，二分类"),
    "electricity": (lambda **kw: load_arff(DATA_DIR / "downloads" / "electricity.arff", **kw), 45312, 8, 2, "电价涨跌，样本量最大"),
}


def load_dataset(name, **kwargs):
    """按注册表名称加载数据集，返回 (X, y, feature_names)。"""
    if name not in DATASETS:
        raise KeyError(f"未知数据集 {name}，可选: {list(DATASETS)}")
    loader = DATASETS[name][0]
    x, y, names, *_ = loader(**kwargs)
    return x, y, names


def list_datasets():
    """打印数据集清单，供选择实验对象。"""
    print(f"{'名称':<12}{'样本':<8}{'特征':<7}{'类别':<7}说明")
    for name, (_, n, d, c, desc) in DATASETS.items():
        print(f"{name:<12}{n:<8}{d:<7}{c:<7}{desc}")


if __name__ == "__main__":
    # 自检与参考统计量：cd code && python data_gen.py
    # 输出的是"有限样本的经验统计量 vs 设定值"的对照，第 4.4 节的验收
    # 需要用到这里打印的经验统计量作为基准。
    x, z = make_synthetic_data(TRUE_MEANS, TRUE_COVS, TRUE_WEIGHTS, n_samples=1500, seed=0)
    emp_pi = np.bincount(z, minlength=3) / len(z)
    emp_mu = np.array([x[z == j].mean(axis=0) for j in range(3)])
    emp_cov = np.array([np.cov(x[z == j].T, bias=True) for j in range(3)])

    print("数据形状:", x.shape)
    print("经验比例:", emp_pi.round(4), " 设定:", TRUE_WEIGHTS)
    print("均值最大偏差 |emp - true| = %.4f" % np.abs(emp_mu - TRUE_MEANS).max())
    print("协方差最大偏差 |emp - true| = %.4f" % np.abs(emp_cov - TRUE_COVS).max())
    for j in range(3):
        ve = np.linalg.eigvalsh(emp_cov[j])
        vt = np.linalg.eigvalsh(TRUE_COVS[j])
        print(
            f"分量 {j}: 经验均值 {emp_mu[j].round(4)} 设定 {TRUE_MEANS[j]}"
            f" | 经验长短轴比 {ve[-1] / ve[0]:.2f} 设定 {vt[-1] / vt[0]:.2f}"
        )
