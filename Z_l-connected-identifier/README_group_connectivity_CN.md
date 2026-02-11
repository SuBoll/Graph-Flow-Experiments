## 群连通性：群值 nowhere-zero 流

本文档说明 **群连通性** 判定脚本及“最小 Z_4 / Z_2×Z_2 区分图”的搜索脚本。当给定具体的有限阿贝尔群 $`A`$ 时，即考虑 **A-连通**（图是否 A-连通）。

**相关文件：**

- `group_connectivity_solver.py`：判定一个连通多重图对给定有限阿贝尔群 $`A`$ 是否 **A-连通**，并对给定的 A-边界 $`\beta`$ 求 **beta-流**（定向 + 流值）。
- `find_smallest_group_connected.py`：在可配置的顶点数范围内枚举简单 2-边连通图，并找出满足“Z_2×Z_2 连通但非 Z_4 连通”或“Z_4 连通但非 Z_2×Z_2 连通”的最小图。（注：该脚本需后续进一步修改后再启用）

**依赖：** Python 3.8+、`networkx`。仅用求解器时不需要 matplotlib/numpy。

---

## 0. 建议的文件放置

将以下文件放在同一目录：

- `group_connectivity_solver.py`
- `find_smallest_group_connected.py`（可选；会导入求解器）
- `README_group_connectivity.md`（英文）
- `README_group_connectivity_CN.md`（本文件，中文）

---

## 1. 数学定义

设 $`A`$ 为有限阿贝尔群，单位元 $`0_A`$；$`G`$ 为连通无向多重图（无自环）。

### 1.1 A-边界（A-boundary）

$`G`$ 上的 **A-边界** 是指映射

$$
\beta : V(G) \to A
$$

且满足各顶点取值在 $`A`$ 中之和为零：

$$
\sum_{v \in V(G)} \beta(v) = 0_A.
$$

### 1.2 无处为零流与 beta-流

先为每条边取定一个定向。对定向边 $`e = (u \to v)`$，**流值** 取为 $`f(e) \in A \setminus \{0_A\}`$（无处为零）。

顶点 $`v`$ 的 **净流** 定义为

$$
\mathrm{net}(v) = \sum_{\text{从 } v \text{ 出发}} f(e) - \sum_{\text{进入 } v} f(e) \in A.
$$

**beta-流** 是指赋值 $`f : E(G) \to A \setminus \{0_A\}`$，使得对每个顶点 $`v`$ 有

$$
\mathrm{net}(v) = \beta(v) \quad \text{（在 } A \text{ 中）}.
$$

（将某条边反向相当于把该边的 $`f(e)`$ 换成 $`-f(e)`$，故定向的选取不影响 beta-流的存在性。）

### 1.3 A-连通

若对 **任意** A-边界 $`\beta`$ 都存在 beta-流，则称 $`G`$ 为 **A-连通**。

### 1.4 代码中的群表示

由 **有限阿贝尔群的分类定理**，任意有限阿贝尔群都同构于若干循环群的直积。因此程序中把 $`A`$ 表示为循环群的直积：

$$
A \cong \mathbb{Z}_{m_1} \times \mathbb{Z}_{m_2} \times \cdots \times \mathbb{Z}_{m_r},\quad m_i \ge 2.
$$

群元为 $`(x_1,\ldots,x_r)`$，$`x_i \in \{0,\ldots,m_i-1\}`$。例如：

- $`[3]`$ → $`A = \mathbb{Z}_3`$，元素 $`0,1,2`$。
- $`[2, 2]`$ → $`A = \mathbb{Z}_2 \times \mathbb{Z}_2`$，元素 $`(0,0), (0,1), (1,0), (1,1)`$。

---

## 2. 快速使用

### 2.1 运行求解器自带的 demo

```bash
python group_connectivity_solver.py
```

会跑内置示例（三角形图、A = Z_3 与 Z_2×Z_2，并对指定 beta 输出一组 beta-流）。

### 2.2 对指定 beta 检查并输出定向与流值

```python
from group_connectivity_solver import GroupConnectivitySolver, _build_multigraph_from_edges

n = 3
edges = [(1, 2), (2, 3), (3, 1)]
Gm = _build_multigraph_from_edges(n, edges)
solver = GroupConnectivitySolver(Gm, group_moduli=[3])

# A-边界：各顶点取值在 Z_3 中之和为 0。例如 1+1+1 = 0 (mod 3)。
beta = {1: (1,), 2: (1,), 3: (1,)}

feasible, sol = solver.check_beta(beta, verbose=True)
if feasible and sol:
    orientation = sol.get_orientation()   # 定向：(tail, head) 的列表
    flow_values = sol.get_flow_values()  # 流值：边编号 -> A 中元素
```

### 2.3 搜索“最小 Z_2×Z_2 与 Z_4 区分图”

在 `find_smallest_group_connected.py` 中修改 `n_min`、`n_max` 后执行：

```bash
python find_smallest_group_connected.py
```

---

## 3. API（group_connectivity_solver.py）

### 3.1 FiniteAbelianGroup

- **`FiniteAbelianGroup(moduli)`**  
  $`\mathrm{moduli}`$ 为元组，如 $`(3)`$ 或 $`(2, 2)`$，各 $`m_i \ge 2`$。

- **`zero()`** — 单位元。
- **`add(a, b)`**、**`neg(a)`** — 群加法与逆元。
- **`elements()`** — 所有群元列表。
- **`is_zero(a)`** — 是否 $`a = 0_A`$。
- **`format(a)`** — 单个群元的字符串（如 Z_2×Z_2 的 $`(1,0)`$）。

### 3.2 FlowSolution

当存在 beta-流时，由 `solve_for_boundary` 返回。

- **`group`**、**`vertices`**、**`edges`**、**`beta`**、**`flow`** — 群、顶点列表、定向边 (tail, head)、边界、流字典（边编号 → 流值）。
- **`get_orientation()`** — 每条边的 $`(\mathrm{tail}, \mathrm{head})`$ 列表。
- **`get_flow_values()`** — 边编号 → $`f(e) \in A \setminus \{0\}`$ 的字典。
- **`pretty_print(max_edges=None)`** — 打印边界、定向与流值。

### 3.3 GroupConnectivitySolver

**`GroupConnectivitySolver(multigraph, group_moduli)`**

- **`multigraph`**：`networkx.MultiGraph`，连通、无自环。
- **`group_moduli`**：如 `[3]` 或 `[2, 2]`。

**方法：**

- **`enumerate_boundaries()`** — 枚举所有 A-边界（和为 $`0_A`$）。
- **`solve_for_boundary(beta)`** — 若存在 nowhere-zero beta-流则返回 `FlowSolution`，否则返回 `None`。
- **`is_A_connected(verbose=False)`** — 图 A-连通时返回 `(True, None)`，否则返回 `(False, 反例 beta)`。
- **`get_all_infeasible_boundaries(verbose=False)`** — 返回所有不存在 beta-流的边界列表（可能很大）。
- **`check_beta(beta, verbose=True, max_edges_in_print=None)`** — 检查一个边界；若可行且 `verbose` 为真则打印定向与流；返回 `(True, solution)` 或 `(False, None)`。

### 3.4 辅助函数

- **`_build_multigraph_from_edges(n, edges)`** — 由顶点 $`1,\ldots,n`$ 及边列表 $`(u,v)`$ 构造 `nx.MultiGraph`。

---

## 4. find_smallest_group_connected.py 说明

该脚本会：

1. 对 $`n \in [\texttt{n\_min}, \texttt{n\_max}]`$ 枚举 **简单**、**2-边连通** 的 $`n`$ 点图。
2. 只保留边数 $`m \ge \min\{\lceil 4n/3\rceil, \lceil 3(n-1)/2\rceil\}`$ 的图。
3. 对每个图判定 Z_2×Z_2 与 Z_4 连通性。若某图是由“已判为 A-连通的图（顶点集相同、少一条边）”加一条边得到，则直接判定为 A-连通，不再调用求解器。
4. 输出满足“Z_2×Z_2 连通但非 Z_4 连通”以及“Z_4 连通但非 Z_2×Z_2 连通”的 **最小** 图。

在 `main()` 开头的 `n_min`、`n_max` 可修改顶点范围。

---

## 5. 性能说明

- 群连通性（对给定群 $`A`$ 的 A-连通性）通过枚举所有 A-边界（共 $`|A|^{n-1}`$ 个，因最后一个顶点由和为零唯一确定）并对每个边界做 nowhere-zero 流的回溯搜索来判定，复杂度关于 $`|V|`$、$`|E|`$ 是指数级的。
- 实现面向 **小图、小群** 的实验；更大规模可考虑理论刻画或专用算法。

