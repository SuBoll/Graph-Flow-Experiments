## SZ_5（奇数模）计算实验代码包：求解器 + 枚举器

本目录用于论文/报告的计算实验与可复现性说明，包含两个脚本（建议放在同一个 GitHub 文件夹中）：

- `szl_odd_solver.py`：奇数模 $l$ 的 $SZ_l$ 判定与 $\beta$-定向求解器（本项目使用 $l=5$）。
- `generate_nonisomorphic_4v12e.py`：枚举满足约束的 4 顶点、12 边（计重）的无向多重图；按同构去重后绘制总览图；并对每个图调用 `szl_odd_solver.py` 进行 $SZ_5$ 检测。

本 README 是**主说明文件**：先整体解释两份代码在实验中的分工，再分别详细解释每个脚本的数学建模与实现流程（包括 DFS 剪枝与同构去重的具体做法），以便审阅与复现。

> 备注：运行时可能看到 `libpng warning: iCCP: known incorrect sRGB profile`。这通常是 PNG 色彩配置相关警告，不影响保存与显示，可忽略。

---

## 0. 文件结构与放置建议

推荐将以下文件放在同一目录（相对路径保持不变）：
 
- `szl_odd_solver.py`
- `generate_nonisomorphic_4v12e.py`
- `README.md`（本文件）
- `README_EN.md`（英文版）

`generate_nonisomorphic_4v12e.py` 会动态导入同目录下的 `szl_odd_solver.py` 来获取 `SZlOddSolver` 类，因此两者需要处于同一目录（或你自行修改导入路径）。

---

## 1. 依赖与安装

- Python 3.8+
- `networkx`
- `matplotlib`
- `numpy`

安装示例：

```bash
python -m pip install networkx matplotlib numpy
```

---

## 2. 快速复现（推荐路径）

### 2.1 运行枚举 + 总览图 + $SZ_5$ 检测

```bash
python generate_nonisomorphic_4v12e.py
```

运行后将：

- 弹出 Matplotlib 总览图窗口（所有非同构代表元的网格图）
- 在“运行脚本时的当前工作目录”保存 PNG 总览图：
  - `nonisomorphic_4v12e_SZ_5_overview.png`
- 在控制台打印 $SZ_5$ 判定结果；若某图不是 $SZ_5$，会打印一个不可行的 witness $\beta$（向量形式）

### 2.2 仅运行求解器示例

```bash
python szl_odd_solver.py
```

脚本内置 `main()`，你可以修改其中的 `mod/n/edges/beta` 来测试不同图与不同边界 $\beta$。

---

## 3. `szl_odd_solver.py`：问题定义（读者视角）

### 3.1 合法边界 $\beta$

给定无向连通多重图 $G$（允许重边、不允许自环）与正奇数 $l$。

称 $\beta:V(G)\to\mathbb{Z}_l$ 为合法边界（boundary），当且仅当：

- $\beta(v)\in\{0,1,\dots,l-1\}$
- $\sum_{v\in V}\beta(v)\equiv 0\pmod l$

后一条件是必要的，因为任意定向都满足 $\sum_v(\text{out}-\text{in})=0$。

### 3.2 $\beta$-定向与 $SZ_l$

对给定 $\beta$，若存在一种对 $G$ 的定向，使得对每个顶点 $v$：

$$
(\operatorname{outdeg}(v)-\operatorname{indeg}(v))\equiv \beta(v)\pmod l
$$

则称该定向为一组 $\beta$-定向。

图 $G$ 称为 $SZ_l$，若对**所有**合法 $\beta$ 都存在 $\beta$-定向。

---

## 4. `szl_odd_solver.py`：数学建模（实现对应）

### 4.1 重边打包：用 $y$ 变量代替逐边方向

对无序点对 $\{u,v\}$（规范化为 $u<v$）若有 $k$ 条平行边，令变量 $y\in\{0,1,\dots,k\}$ 表示其中有 $y$ 条边定向为 $u\to v$。则：

- $u$ 点的净贡献（out-in）为 $y-(k-y)=2y-k$
- $v$ 点的净贡献为 $-(2y-k)$

因此每个点对只需搜索一个整数变量 $y$，而不是枚举 $2^k$ 种逐边方向。

### 4.2 符号约定与常数项 $C_v$

对边束 $e=(u,v)$（规范化 $u<v$）定义符号：

- $\operatorname{sign}(u,e)=+1$
- $\operatorname{sign}(v,e)=-1$

对每个顶点定义：

$$
S(v)=\sum_{e\ni v}\operatorname{sign}(v,e)\cdot y_e
$$

把 $(2y_e-k_e)$ 展开后，可写成：

$$
C_v+2S(v)\equiv \beta(v)\pmod l,\qquad
C_v=\sum_{e\ni v}\operatorname{sign}(v,e)\cdot(-k_e)
$$

### 4.3 奇数模的关键：消去系数 2

当 $l$ 为奇数时，2 在 $\mathbb{Z}_l$ 中可逆。记 $2^{-1}$ 为其逆元，则：

$$
S(v)\equiv 2^{-1}(\beta(v)-C_v)\pmod l
$$

代码中把右端预计算为 `target_residue[v_idx]`，后续 DFS 的目标就是让每个顶点的 $S(v)$ 落在对应同余类。

---

## 5. `szl_odd_solver.py`：实现流程（按函数）

### 5.1 核心数据结构

- `EdgeBundle(u, v, k)`：记录一个无序点对及重数 $k$（并保证 `u < v`）。
- `OddOrientationSolution`：保存一组解，包括：
  - `y_by_pair[(u,v)]`：每个点对的 $y$
  - `out_minus_in[v]`：每个顶点的整数 out-in（未取模）
  - `directions`：逐条边展开后的方向 `(tail, head)`（用于复现与核验）

### 5.2 `SZlOddSolver.__init__`：预处理

初始化完成以下工作：

- **输入校验**：$l$ 为正奇数；图无自环；图连通
- **顶点与索引**：`vertices = sorted(nodes)`，`index_of_vertex[v] = idx`
- **统计边束**：`_collect_edge_bundles()` 统计每个点对的重数 $k$
- **构造符号结构**：`_build_signs()` 为每个顶点保存其关联边束及符号 $\pm1$
- **度数**：按重数计算每个顶点度
- **常数项**：预计算 `C_vec[idx] = C_v`
- **逆元**：预计算 `inv2 = 2^{-1} (mod l)`

### 5.3 `enumerate_betas()`：枚举所有合法 $\beta$

设顶点数为 $n$。实现枚举方式为：

- 枚举前 $n-1$ 个顶点的取值（每个在 $0,\dots,l-1$），共 $l^{n-1}$ 种
- 最后一个顶点的取值由 $\sum\beta\equiv 0\pmod l$ 唯一确定

这样避免了 $l^n$ 的完全枚举。

### 5.4 `solve_for_beta(beta)`：DFS + 剪枝（核心）

整体流程：

1. **合法性检查**：顶点集合匹配；取值范围正确；总和为 0（mod $l$）
2. **计算目标同余类**：对每个顶点计算 `target_residue[v]`
3. **建立变量域**：对每条边束 $e$，$y_e\in\{0,\dots,k_e\}$
4. **变量排序**：按域大小从小到大（即 $k_e$ 小的优先）
5. **DFS 搜索**：
   - 状态：`partial_sum[v]`（当前已赋值变量贡献的 $S(v)$ 整数部分和）
   - 递归：对当前边束尝试每个 $y$ 值，更新两个端点的 `partial_sum`
6. **剪枝（区间 + 同余可达性）**：
   - 对每个顶点 $v$，计算剩余未赋值边束对 $S(v)$ 的可达区间 $[L,U]$
   - 令 `need ≡ target_residue[v] - partial_sum[v] (mod l)`，判断是否存在 $t\in[L,U]$ 使得
     $t\equiv \text{need}\pmod l$
   - 等价判断：是否存在整数 $q$ 使 $L\le \text{need}+ql\le U$
   - 若不存在则剪枝回溯
7. **终止与组装**：
   - 全部变量赋值后检查所有顶点同余是否满足
   - 将每个边束的 $y$ 展开为逐边方向（前 $y$ 条为 $u\to v$，后 $k-y$ 条为 $v\to u$）
   - 最后校验 `out_minus_in[v] % l == beta[v]`

### 5.5 `is_SZl(verbose=False, max_beta=None)`：判定与 witness

通过枚举所有合法 $\beta$ 并逐个调用 `solve_for_beta(beta)`：

- 一旦遇到不可行 $\beta$，返回 `(False, beta)` 作为 witness（反例）
- 否则返回 `(True, None)`

`max_beta` 可用于调试时截断枚举。

---

## 6. `generate_nonisomorphic_4v12e.py`：总体功能

该脚本把“构造实验数据集 + 可视化 + 批量判定”合并在一起，主要任务是：

1. 枚举满足约束的 4 顶点、12 边（计重）多重图
2. 在顶点置换意义下去掉同构重复（保留代表元）
3. 绘制所有代表元的总览图，并保存 PNG
4. 对每个代表元调用 `SZlOddSolver(..., 5)` 判定是否为 $SZ_5$，并在必要时打印 witness $\beta$

---

## 7. `generate_nonisomorphic_4v12e.py`：实现流程（按模块）

### 7.1 枚举空间定义（精确）

顶点固定为 $\{1,2,3,4\}$，只考虑 6 个无序点对：

$$
(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
$$

对每对分配重数 $k_{uv}\in\{0,1,2,3\}$，并筛选：

- 总边数（计重）：$\sum k_{uv}=12$
- 最小度：每个顶点度（计重）$\ge 4$
- 连通性：把 $k_{uv}>0$ 当作简单边，要求连通

### 7.2 同构去重：canonical key（4! 全置换）

脚本实现了一个确定性 canonicalization：

- 枚举所有 $4!$ 顶点置换
- 对每个置换计算重标号后的 6 元组表示
- 取字典序最小者作为 canonical key
- 用字典按 key 去重，得到非同构代表元列表

### 7.3 绘图：保证每个子图视觉尺度一致

多重边用弯曲弧线绘制时，Matplotlib 可能对不同子图自动缩放，造成“有的图看起来更大/更小”。脚本通过：

- 固定径向布局半径 `radius`
- 对每个子图固定 `xlim/ylim` 为同一范围

来保证视觉尺度一致。

### 7.4 保存总览图（PNG）

脚本会保存总览图到当前工作目录（即你运行 `python ...` 时所在目录）：

- `nonisomorphic_4v12e_SZ_5_overview.png`

并在控制台打印实际保存路径。

### 7.5 批量 $SZ_5$ 判定与输出

对每个代表元：

- 构造 `nx.MultiGraph`
- `solver = SZlOddSolver(Gm, 5)`
- `ok, witness = solver.is_SZl(verbose=False)`

若 `ok` 为 False，则打印：

- 图编号
- 6 个点对的边重数
- 顶点度
- witness $\beta$（向量形式）：
  - `[witness[v] for v in solver.vertices]`

---

## 8. 性能与可复现性提示

- **求解器复杂度**：合法 $\beta$ 的数量为 $l^{n-1}$；单个 $\beta$ 的 DFS 最坏指数级。本实现主要面向中小规模实例的实验验证与复现。
- **确定性**：
  - `szl_odd_solver.py`：顶点与边束排序固定；同输入应给出一致的搜索顺序与输出（若存在解）
  - `generate_nonisomorphic_4v12e.py`：同构去重采用全置换 canonical key，完全确定


