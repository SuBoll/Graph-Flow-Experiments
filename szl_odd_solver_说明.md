# SZ_l 判定与求解器（奇数模版本）使用说明

## 一、概述

本代码实现了一个针对**奇数模 l** 的 SZ_l 图性质判定与 beta-定向求解器。该求解器可以：

1. **判定一个图是否为 SZ_l**：枚举所有合法的 beta 函数，检查是否每个 beta 都存在对应的定向方案。
2. **求解给定 beta 的定向**：对于用户指定的 beta 函数，输出一组具体的边定向方案。
3. **输出反例**：如果图不是 SZ_l，会输出一个不可行的 beta 作为反例。

**重要限制**：本求解器**仅支持奇数模 l**（即 l 必须是正奇数，如 3, 5, 7, 9, ...）。对于偶数模的情况，请使用其他版本的求解器。

---

## 二、数学背景

### 2.1 基本定义

给定一个**无向连通多重图** G（允许重边，但不允许自环），以及一个**正奇数 l**。

#### Z_l-boundary（Z_l 边界）

一个函数 $\beta: V(G) \to \mathbb{Z}_l$ 称为 **Z_l-boundary**，如果满足：
- 对每个顶点 $v$，$\beta(v) \in \{0, 1, 2, \ldots, l-1\}$
- 所有顶点的 beta 值之和在模 l 意义下为 0：$\sum_{v \in V} \beta(v) \equiv 0 \pmod{l}$

#### beta-定向（beta-orientation）

对于给定的图 G 和 Z_l-boundary $\beta$，如果能找到一个**定向**（给每条边指定一个方向），使得对每个顶点 $v$：
$$(\text{出度} - \text{入度}) \equiv \beta(v) \pmod{l}$$
则称这个定向为**beta-定向**。

#### SZ_l 性质

图 G 称为 **SZ_l**，如果对于**所有可能的 Z_l-boundary $\beta$**，都存在对应的 beta-定向。

### 2.2 多重边的处理

对于两个顶点 $u$ 和 $v$ 之间的 $k$ 条重边，我们采用以下方式处理：

- 设在这 $k$ 条边中，有 $y$ 条边定向为 $u \to v$（其中 $0 \leq y \leq k$）
- 则剩余的 $k-y$ 条边定向为 $v \to u$
- 顶点 $u$ 的贡献为：$(y) - (k-y) = 2y - k$
- 顶点 $v$ 的贡献为：$(k-y) - (y) = -(2y - k)$

因此，对于重数为 $k$ 的边对，顶点 $u$ 的贡献可以是集合 $\{k, k-2, k-4, \ldots, -k+2, -k\}$ 中的任意值（在模 l 意义下）。

### 2.3 约束转换

对于每个顶点 $v$，我们需要满足：
$$C_v + 2 \sum_{e \ni v} \text{sign}(v,e) \cdot y_e \equiv \beta(v) \pmod{l}$$

其中：
- $C_v = \sum_{e \ni v} \text{sign}(v,e) \cdot (-k_e)$ 是常数项
- $y_e$ 是边 $e$ 的变量（表示该边对中有多少条边定向为某个方向）
- $\text{sign}(v,e)$ 表示顶点 $v$ 在边 $e$ 上的符号（+1 或 -1）

由于 $l$ 是奇数，2 在 $\mathbb{Z}_l$ 中可逆，我们可以将约束转换为：
$$\sum_{e \ni v} \text{sign}(v,e) \cdot y_e \equiv \gamma(v) \pmod{l}$$

其中 $\gamma(v) = 2^{-1} \cdot (\beta(v) - C_v) \bmod l$。

---

## 三、算法原理

### 3.1 求解流程

#### 步骤 1：预处理
1. 收集所有边对及其重数（`_collect_edge_bundles`）
2. 构建每个顶点关联的边及其符号（`_build_signs`）
3. 计算常数项 $C_v$ 和 2 的逆元

#### 步骤 2：枚举 beta（用于判定 SZ_l）
- 枚举前 $n-1$ 个顶点的 beta 值（每个在 $0..l-1$ 范围内）
- 最后一个顶点的 beta 值由总和为 0 的条件确定
- 总共有 $l^{n-1}$ 个不同的 beta 需要检查

#### 步骤 3：对每个 beta 求解定向（回溯搜索）
使用深度优先搜索（DFS）来寻找满足模约束的 $y_e$ 值：

1. **变量定义**：对每条边 $e$（重数为 $k_e$），变量 $y_e \in \{0, 1, 2, \ldots, k_e\}$
2. **搜索顺序**：按边的重数从小到大排序（优先处理重数小的边，提高剪枝效率）
3. **剪枝策略**：
   - 对每个顶点 $v$，计算剩余未分配边的可能贡献范围 $[L, U]$
   - 检查是否存在整数 $q$ 使得：$L \leq \gamma(v) - \text{当前部分和} + q \cdot l \leq U$
   - 如果不存在这样的 $q$，则剪枝
4. **回溯**：当找到一个解时返回，否则回溯尝试下一个 $y_e$ 值

#### 步骤 4：组装解
找到所有 $y_e$ 后，构造具体的边定向：
- 对每条边对 $(u,v)$（重数 $k$），前 $y$ 条边定向为 $u \to v$，后 $k-y$ 条边定向为 $v \to u$
- 验证每个顶点的 (出度 - 入度) 在模 l 意义下等于 $\beta(v)$

### 3.2 复杂度分析

- **beta 枚举**：$O(l^{n-1})$，其中 $n$ 是顶点数
- **单个 beta 求解**：最坏情况是指数级（回溯搜索），但通过剪枝可以大幅减少搜索空间
- **总体复杂度**：$O(l^{n-1} \cdot \text{回溯搜索时间})$，对于小规模图（顶点数 $\leq 10$，$l \leq 10$）通常可接受

---

## 四、代码结构

### 4.1 主要类

#### `EdgeBundle`
表示一个无向边对及其重数：
- `u`, `v`：顶点（保证 $u < v$）
- `k`：重数（该顶点对之间的边数）

#### `OddOrientationSolution`
表示一个 beta-定向解：
- `modulus`：模数 $l$
- `vertices`：顶点列表
- `edge_bundles`：边对列表
- `y_by_pair`：字典，`y_by_pair[(u,v)]` 表示边对 $(u,v)$ 中有多少条边定向为 $u \to v$
- `out_minus_in`：字典，`out_minus_in[v]` 表示顶点 $v$ 的 (出度 - 入度) 的整数值
- `beta`：给定的 beta 函数
- `directions`：逐条边的定向列表，每个元素是 `(tail, head)`

#### `SZlOddSolver`
核心求解器类：

**初始化方法**：
- `__init__(multigraph, modulus)`：初始化求解器
  - `multigraph`：NetworkX 的 MultiGraph 对象
  - `modulus`：模数 $l$（必须是正奇数）

**主要方法**：
- `enumerate_betas()`：生成器，枚举所有合法的 beta 函数
- `solve_for_beta(beta)`：对给定的 beta 求解定向
  - 返回：`(是否可行, 解对象或None)`
- `is_SZl(verbose=False, max_beta=None)`：判定图是否为 SZ_l
  - `verbose`：是否输出进度信息
  - `max_beta`：可选，限制检查的 beta 数量（用于调试）
  - 返回：`(是否为SZ_l, 反例beta或None)`

**辅助方法**：
- `_collect_edge_bundles()`：收集边对及其重数
- `_build_signs()`：构建每个顶点关联的边及其符号
- `_compute_degrees()`：计算每个顶点的度数

### 4.2 辅助函数

- `build_graph_from_edges(n, edges)`：从边列表构建多重图
  - `n`：顶点数（顶点编号从 1 到 n）
  - `edges`：边列表，每个元素是 `(u, v)` 元组

---

## 五、使用方法

### 5.1 基本使用

#### 示例 1：判定图是否为 SZ_l

```python
import networkx as nx
from szl_odd_solver import SZlOddSolver, build_graph_from_edges

# 定义图：4个顶点，边列表
n = 4
edges = [(1, 2)]*3 + [(1, 3)]*3 + [(1, 4)]*3 + [(2, 3), (3, 4), (4, 2)]

# 构建图
Gm = build_graph_from_edges(n, edges)

# 创建求解器（l=5，必须是奇数）
solver = SZlOddSolver(Gm, modulus=5)

# 判定是否为 SZ_5
is_sz, witness = solver.is_SZl(verbose=True)

if is_sz:
    print(f"该图是 SZ_5")
else:
    print(f"该图不是 SZ_5")
    print(f"反例 beta: {[witness[v] for v in solver.vertices]}")
```

#### 示例 2：求解给定 beta 的定向

```python
# 定义 beta（必须在 0..l-1 且总和 ≡ 0 mod l）
beta = {1: 0, 2: 1, 3: 2, 4: 2}  # 总和 = 5 ≡ 0 (mod 5)

# 求解
ok, sol = solver.solve_for_beta(beta)

if ok:
    print("可行！")
    sol.pretty_print()  # 打印详细的定向信息
else:
    print("不可行")
```

### 5.2 自定义图

你可以使用 NetworkX 的 MultiGraph 直接构建图：

```python
import networkx as nx

Gm = nx.MultiGraph()
Gm.add_nodes_from([1, 2, 3, 4])
Gm.add_edge(1, 2)
Gm.add_edge(1, 2)  # 添加重边
Gm.add_edge(1, 3)
Gm.add_edge(2, 3)
Gm.add_edge(3, 4)

solver = SZlOddSolver(Gm, modulus=3)
```

### 5.3 运行示例程序

直接运行 `szl_odd_solver.py` 文件：

```bash
python szl_odd_solver.py
```

程序会执行 `main()` 函数中的示例，你可以修改 `main()` 函数中的参数来测试不同的图和模数。

---

## 六、输出说明

### 6.1 `is_SZl()` 的输出

- 如果图是 SZ_l：返回 `(True, None)`
- 如果图不是 SZ_l：返回 `(False, beta_dict)`，其中 `beta_dict` 是一个不可行的 beta 反例

### 6.2 `solve_for_beta()` 的输出

- 如果可行：返回 `(True, OddOrientationSolution对象)`
- 如果不可行：返回 `(False, None)`

### 6.3 `pretty_print()` 的输出格式

调用 `sol.pretty_print()` 会输出：

```
—— One beta-orientation solution (odd l) ——
l=5
beta (vector): [0, 1, 2, 2]
Vertex out-in and check (mod l):
  v=1: out-in=0, mod 5 = 0
  v=2: out-in=1, mod 5 = 1
  v=3: out-in=2, mod 5 = 2
  v=4: out-in=2, mod 5 = 2
Pairs (u,v), k, y(u->v), contribution to u (2y-k):
  (1,2), k=3, y=2, 2y-k=1
  (1,3), k=3, y=1, 2y-k=-1
  (1,4), k=3, y=0, 2y-k=-3
  (2,3), k=1, y=0, 2y-k=-1
  (2,4), k=1, y=1, 2y-k=1
  (3,4), k=1, y=1, 2y-k=1
Per-edge directions (first few):
  e1: 1->2
  e2: 1->2
  e3: 2->1
  ...
```

---

## 七、注意事项

### 7.1 限制条件

1. **模数必须是奇数**：本求解器仅支持奇数模 $l$。如果传入偶数模，会抛出 `ValueError`。
2. **图必须连通**：如果图不连通，会抛出 `ValueError`。
3. **不允许自环**：如果图中存在自环，会抛出 `ValueError`。

### 7.2 性能考虑

1. **顶点数**：建议顶点数 $\leq 10$，否则 beta 枚举数量会非常大（$l^{n-1}$）。
2. **模数大小**：建议 $l \leq 10$，否则枚举和搜索时间会显著增加。
3. **边数**：边数过多会导致回溯搜索变慢，但通常影响较小。

### 7.3 beta 函数的合法性

在调用 `solve_for_beta(beta)` 时，必须确保：
- `beta` 的键集合等于图的顶点集合
- 每个 `beta[v]` 在 $[0, l-1]$ 范围内
- `sum(beta.values()) % l == 0`

如果不满足这些条件，函数会返回 `(False, None)`。

### 7.4 调试技巧

如果图很大或 $l$ 很大，可以使用 `max_beta` 参数限制检查的 beta 数量：

```python
is_sz, witness = solver.is_SZl(verbose=True, max_beta=100)
```

这样只会检查前 100 个 beta，用于快速测试。

---

## 八、算法优化说明

### 8.1 剪枝策略

代码中使用了多种剪枝策略来提高搜索效率：

1. **按重数排序**：优先处理重数小的边，因为这些边的搜索空间更小，更容易剪枝。
2. **范围检查**：对每个顶点，计算剩余边的可能贡献范围，如果无法达到目标模值，则剪枝。
3. **模约束检查**：在搜索过程中持续检查模约束，一旦发现不可行立即回溯。

### 8.2 数学优化

1. **2 的逆元预计算**：由于 $l$ 是奇数，2 在 $\mathbb{Z}_l$ 中可逆，我们预先计算 $2^{-1} \bmod l$，避免重复计算。
2. **常数项预计算**：$C_v$ 在初始化时计算一次，后续求解时直接使用。

---

## 九、常见问题

### Q1: 为什么只支持奇数模？

A: 当 $l$ 为奇数时，2 在 $\mathbb{Z}_l$ 中可逆，这使得约束转换变得简单。对于偶数模的情况，需要更复杂的处理，本代码未实现。

### Q2: 如果图不是 SZ_l，如何理解反例？

A: 反例 beta 是一个 Z_l-boundary，它满足所有合法性条件（值域正确、总和为 0），但不存在对应的 beta-定向。这证明了图不是 SZ_l。

### Q3: 如何验证输出的定向是否正确？

A: 可以手动检查：
1. 对每个顶点 $v$，计算其出度减去入度
2. 检查这个值在模 $l$ 意义下是否等于 $\beta(v)$
3. 检查所有顶点的 beta 值之和是否在模 $l$ 意义下为 0

代码中的 `pretty_print()` 方法已经包含了这些验证信息。

### Q4: 可以处理有向图吗？

A: 不可以。本代码假设输入是无向多重图，然后寻找一个定向方案。如果输入已经是有向图，需要先转换为无向图。

---

## 十、扩展阅读

### 相关概念

- **SZ_l 性质**：这是图论中的一个重要性质，与图的连通性、度数分布等密切相关。
- **beta-定向**：这是一种特殊的图定向问题，在组合优化和理论计算机科学中有广泛应用。
- **模约束系统**：本问题本质上是一个模约束的整数规划问题。

### 进一步优化方向

1. **并行化**：可以对不同的 beta 并行求解，提高速度。
2. **更智能的剪枝**：可以使用更高级的约束传播技术。
3. **启发式搜索**：对于大规模问题，可以使用启发式方法快速找到解或证明不可行。

---

## 十一、版本信息

- **版本**：1.0
- **最后更新**：2025年
- **依赖库**：Python 3.7+, NetworkX

---

## 十二、联系与反馈

如有问题或建议，请通过适当渠道反馈。
