# `train.py` 的语法结构与 PyTorch 对象

> 对应文件：`src/train.py`（257 行）
>
> 目标：看懂每个 `torch` / `nn` 对象是什么、训练循环为什么长这样，以及这些函数之间怎么配合。

## 0. 一张图看懂整个文件

```text
cross_validate(X, y, cfg)
│
├─ stratified_kfold(y)              把 351 个样本分成 5 折
├─ for 每一折:
│   ├─ standardize(X[tr], X[te])    只用训练折的均值/方差
│   ├─ train_fold(...)             训练这一折的模型
│   │   ├─ MLP(...)                34 → 隐藏层 → 1
│   │   ├─ _optimizer(model, cfg)  SGD / Adam
│   │   ├─ IonDataset + DataLoader 把数组变成 batch
│   │   ├─ for epoch:
│   │   │   └─ for xb, yb in loader:
│   │   │       zero_grad → forward → loss → backward → step
│   │   └─ evaluate(...)           在训练集/验证集上算 loss 和概率
│   └─ binary_metrics(...)         算 acc / precision / recall / F1
└─ 汇总 5 折的均值和标准差
```

`train.py` 本身不定义网络，网络在 `model.py`；它只负责“怎么训练、怎么评估、怎么交叉验证”。

## 1. `torch.nn` 到底是什么

`torch.nn` 是 PyTorch 的**神经网络零件库**，里面有两类东西：

1. **有参数的层**：`nn.Linear`、`nn.BatchNorm1d`。它们内部存着可训练的 `weight` / `bias`。
2. **无参数的层/函数包装**：`nn.ReLU`、`nn.Dropout`、`nn.Sigmoid`。它们只做计算。
3. **损失函数**：`nn.BCEWithLogitsLoss`、`nn.CrossEntropyLoss`。它们也是 `nn.Module`，可以像函数一样调用。

最重要的是 `nn.Module`：所有网络、层、损失都继承它。

```python
class MLP(nn.Module):          # 继承 nn.Module
    def __init__(self):
        super().__init__()      # 必须先初始化父类
        self.fc = nn.Linear(34, 64)   # 赋值给 self.*，才会被注册成参数

    def forward(self, x):       # 定义“数据怎么流动”
        return self.fc(x)
```

`nn.Module` 帮你做了四件事：

- **参数注册**：`self.fc = nn.Linear(...)` 后，`model.parameters()` 能自动找到 `fc.weight` / `fc.bias`；
- **调用转发**：写 `model(x)` 时，实际调用的是 `model.forward(x)`，但还会处理 hook；
- **模式切换**：`model.train()` / `model.eval()` 控制 Dropout、BatchNorm 的行为；
- **状态保存**：`model.state_dict()` 返回 `{参数名: tensor}`，可保存/加载。

### `nn.Linear` 是什么

`nn.Linear(34, 64)` 内部有：

```text
weight: (64, 34)
bias:   (64,)
```

前向计算是：

```text
y = x @ weight.T + bias
```

输入 `(B, 34)`，输出 `(B, 64)`。它就是一个“带两个张量的结构体 + 一个 operator()”。

### `nn.BCEWithLogitsLoss` 是什么

二分类的交叉熵。名字里的 **WithLogits** 表示：

- 它期望模型输出 **logit**（没有经过 Sigmoid 的原始分数）；
- 它内部先做 Sigmoid，再算 BCE；
- 这样比“先 Sigmoid 再 BCELoss”数值更稳定。

所以 `model.py` 的 `forward` 最后**不接 Sigmoid**，只输出一个 logit；推理时才手动 `torch.sigmoid(logits)`。

### `torch.nn` 和 `torch.nn.functional` 的区别

- `nn.Linear`、`nn.ReLU` 是**对象**，有状态（参数）或可以放进 `nn.Sequential`；
- `torch.nn.functional.relu(x)` 是**纯函数**，没有状态；
- `nn.ReLU()` 内部基本就是调用 `F.relu`。

初学阶段记住：**有参数用 `nn.Xxx`，纯计算可以用 `F.xxx`；本实验统一用 `nn.Xxx`，因为要放进模型结构里。**

## 2. `train.py` 的四个层次

### 2.1 `Config`：超参数的数据类

```python
@dataclass
class Config:
    hidden_dims: tuple[int, ...] = (64, 64)
    activation: str = "relu"
    ...
```

`@dataclass` 自动生成 `__init__`、`__repr__`、`__eq__`。你不用手写：

```python
def __init__(self, hidden_dims=(64,64), ...):
    self.hidden_dims = hidden_dims
```

这就是一个“带默认值的结构体”。`cfg.hidden_dims` 是普通属性访问。

### 2.2 指标函数：不涉及 PyTorch

`binary_metrics(y_true, prob)` 只做 NumPy 运算：

```python
pred = (prob >= 0.5).astype(np.int64)
tp = int(((pred == 1) & (y_true == 1)).sum())
```

它和 `torch` 无关。PyTorch 只负责“产出概率”，指标用 NumPy 算，边界清晰。

### 2.3 评估函数：`@torch.no_grad()` + `model.eval()`

```python
@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    loader = DataLoader(IonDataset(X, np.zeros(len(X))), batch_size=256)
    outs = []
    for xb, _ in loader:
        outs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outs)
```

三个语法点：

- `@torch.no_grad()`：这个装饰器让函数内的计算**不建计算图**，省内存、速度快；评估不需要梯度。
- `model.eval()`：把 Dropout 关掉、BatchNorm 用 running stats；**它不会关闭梯度**，所以还要配 `no_grad`。
- `.cpu().numpy()`：Tensor 转 NumPy。GPU 上的 Tensor 必须先 `.cpu()`。

### 2.4 `train_fold`：核心训练循环

```python
torch.manual_seed(seed)
model = MLP(...)
criterion = nn.BCEWithLogitsLoss()
opt = _optimizer(model, cfg)
train_ds = IonDataset(X_train, y_train)
loader = DataLoader(train_ds, batch_size=..., shuffle=True, ...)

for epoch in range(1, cfg.epochs + 1):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()
```

这七行就是 PyTorch 训练的“固定语法”。下面逐行解释。

## 3. 训练循环七行逐行解释

### 第 1 行：`model.train()`

把模型切到训练模式：

- Dropout 开始随机丢神经元；
- BatchNorm 用当前 batch 的均值和方差，并更新 running stats。

它只改“模式标志”，不改变参数，也不改变梯度。

### 第 2 行：`for xb, yb in loader`

`DataLoader` 是一个可迭代对象，每次产出一个 batch：

- `xb`：`(batch_size, 34)` 的 float Tensor；
- `yb`：`(batch_size,)` 的 float Tensor。

`shuffle=True` 每个 epoch 重新打乱；`generator` 固定随机种子。

### 第 3 行：`opt.zero_grad()`

PyTorch 的梯度是**累加**的：`loss.backward()` 会把新的梯度加到 `.grad` 上，而不是覆盖。

所以每个 batch 开头必须清空：

```python
opt.zero_grad()
```

忘记这一行，梯度会越来越大，训练直接崩。

### 第 4 行：`logits = model(xb)`

`model` 是 `nn.Module`，`model(xb)` 不是普通函数调用，而是：

```text
__call__ → forward(xb) → 返回 logits
```

不要写成 `model.forward(xb)`。直接调用 `model(...)` 才会走 PyTorch 注册的 hook 和 `__call__` 逻辑。

输出形状是 `(batch_size,)`，因为 `model.py` 最后做了 `squeeze(-1)`。

### 第 5 行：`loss = criterion(logits, yb)`

`criterion` 是 `nn.BCEWithLogitsLoss()`，它也是一个 `nn.Module`，所以可以像函数一样调用。

它期望：

- `logits`：`(B,)`，原始分数；
- `yb`：`(B,)`，0/1 的 float。

返回一个标量 Tensor（0 维）。

### 第 6 行：`loss.backward()`

反向传播。autograd 从 `loss` 出发，沿计算图把梯度写到每个参数的 `.grad`：

```text
fc1.weight.grad
fc1.bias.grad
fc2.weight.grad
...
```

这一行只**算梯度**，不更新参数。

### 第 7 行：`opt.step()`

优化器根据每个参数的 `.grad` 更新参数：

```text
SGD:  w ← w - lr * grad
Adam: 用一阶/二阶动量更新
```

优化器在构造时已经拿到了 `model.parameters()` 的引用，所以 `step()` 改的就是模型里的同一批 Tensor。

## 4. `nn.Module` 的几个常用语法

### `model.parameters()`

返回一个迭代器，产出所有 `requires_grad=True` 的参数。优化器就是靠它知道“要更新谁”。

```python
torch.optim.SGD(model.parameters(), lr=0.01)
```

### `model.state_dict()` / `load_state_dict()`

`state_dict()` 返回一个字典：

```python
{
  "net.0.weight": tensor(...),
  "net.0.bias": tensor(...),
  ...
}
```

它包含参数和 buffer（例如 BatchNorm 的 running mean/var）。

```python
best_state = copy.deepcopy(model.state_dict())
...
model.load_state_dict(best_state)
```

为什么用 `deepcopy`？`state_dict()` 里的 Tensor 和模型参数共享内存；不深拷贝，训练继续更新后“最佳状态”也会被改掉。

### `model.train()` / `model.eval()`

```python
model.train()   # 训练：Dropout 开，BN 用 batch 统计
model.eval()    # 评估：Dropout 关，BN 用 running stats
```

它们**不改变参数**，只改变层的行为。评估时还必须加 `torch.no_grad()`，否则仍会建计算图。

### `nn.Sequential`

`model.py` 里用 `nn.Sequential(*layers)` 把层按顺序串起来：

```python
self.net = nn.Sequential(
    nn.Linear(34, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
```

`self.net(x)` 等价于 `nn.Linear → ReLU → nn.Linear` 依次调用。`*layers` 是列表解包，把列表里的每个层作为独立参数传给 `Sequential`。

## 5. Tensor 与 NumPy 的边界

`train.py` 里频繁出现：

```python
float(loss.detach())
torch.sigmoid(logits).cpu().numpy()
torch.as_tensor(X, dtype=torch.float32)
```

- `loss` 是 0 维 Tensor，带梯度；`float(loss)` 会警告，`loss.detach()` 先脱离计算图，再转 Python float。
- `.cpu()` 把 GPU Tensor 移到 CPU；`.numpy()` 转 NumPy，两者共享内存。
- `torch.as_tensor` 把 NumPy 数组转 Tensor，不复制（如果 dtype 一致）；`torch.tensor` 总是复制。
- `IonDataset.__getitem__` 返回 `(Tensor, Tensor)`，`DataLoader` 再拼成 batch。

## 6. 交叉验证的“胶水”语法

```python
for k, (tr_idx, te_idx) in enumerate(folds):
    X_tr, X_te, _, _ = standardize(X[tr_idx], X[te_idx])
    model, history, best_epoch = train_fold(
        X_tr, y[tr_idx], X_te, y[te_idx], cfg, seed=cfg.seed + k,
        track_history=keep_history and k == 0,
    )
```

- `folds` 是 `list[tuple[train_idx, test_idx]]`，`enumerate` 同时拿到折号和 `(tr_idx, te_idx)`；
- `X[tr_idx]` 是 NumPy 花式索引，按整数数组取行；
- `standardize` 返回四个值，用 `_, _` 忽略均值和标准差；
- `track_history=keep_history and k == 0`：只给第一折记录逐 epoch 曲线，节省时间；
- `cfg.seed + k`：每折用不同但可复现的种子。

最后：

```python
"closed_acc_mean": float(np.mean(closed_accs)),
"open_acc_std": float(np.std(open_accs)),
```

`np.std` 默认除以 5（总体标准差），所以报告里的标准差是 5 折的总体标准差。

## 7. 用 C++ 类比

| PyTorch | C++ 里类似的东西 |
|---|---|
| `nn.Module` | 抽象基类，`forward` 是虚函数；`__call__` 是非虚包装 |
| `nn.Linear` | 带 `weight`、`bias` 两个成员和 `operator()` 的 struct |
| `model.parameters()` | 返回指向所有参数 Tensor 的指针列表 |
| `optimizer` | 持有这些指针的对象，`step()` 原地修改参数 |
| `state_dict()` | `std::map<std::string, Tensor>` |
| `DataLoader` | 按 batch 切数组的迭代器 |

## 8. 最小可运行片段

去掉交叉验证和指标，只留最核心的语法：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(351, 34)
y = torch.randint(0, 2, (351,)).float()

model = nn.Sequential(
    nn.Linear(34, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

for epoch in range(10):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        logits = model(xb).squeeze(-1)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()

model.eval()
with torch.no_grad():
    prob = torch.sigmoid(model(X).squeeze(-1))
```

把这段和 `train.py` 对照，`train_fold` 只是多了：种子、早停、训练曲线、验证集评估。

## 9. 常见错误

1. **忘记 `opt.zero_grad()`**：梯度累加，loss 爆炸。
2. **用 `nn.BCELoss` 但模型输出 logits**：BCE 期望概率，不是 logit；要么模型加 Sigmoid，要么改用 `BCEWithLogitsLoss`。
3. **评估时忘记 `model.eval()`**：Dropout 还在随机丢，BatchNorm 用 batch 统计，结果不稳定。
4. **评估时忘记 `torch.no_grad()`**：能跑，但建图浪费内存，还可能 OOM。
5. **直接改 `state_dict` 里的 Tensor**：它和模型共享内存；保存最佳状态要 `deepcopy`。
6. **把 `loss` 直接 `float(loss)`**：会带梯度警告；用 `loss.detach()` 或 `loss.item()`。
7. **在 `forward` 里做 `.numpy()`**：Tensor 转 NumPy 会断梯度；训练路径里不要转。
8. **以为 `model.eval()` 会停止梯度**：不会，它只改 Dropout/BN 模式。

## 10. 一句话总结

`train.py` 的语法可以压缩成一句：

> **`nn.Module` 定义“算什么”，`DataLoader` 定义“怎么喂”，`loss.backward()` 算梯度，`optimizer.step()` 改参数，`torch.no_grad()` 只做评估。**

把这四件事分清，`train.py` 就不难读了。
