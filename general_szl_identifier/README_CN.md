## $SZ_l$（一般模）计算实验代码包：求解器 + 枚举器

本目录包含针对**一般模 $l$**（奇数或偶数）的 $SZ_l$ 判定与枚举脚本，与奇数模版本（`szl_odd_solver.py` + `generate_nonisomorphic_4v_szl.py`）互补：

- `szl_solver.py`：一般模 $l$（任意正整数）的 $SZ_l$ 判定与 $\beta$-定向求解器。
- `generate_nonisomorphic_szl.py`：对**给定的 $n$、$`m`$、$`l`$**，枚举 $n$ 顶点、$`m`$ 条边（计重）的无向多重图（约束：每点最小度 $\ge l-1$，每对顶点边重数 $\le l-2$，边连通度 $\ge 2$，连通）；排除含已知 $SZ_l$ 子图（3 顶点或 4 顶点）的图；按同构去重后绘制总览图；对每个图调用求解器进行 $SZ_l$ 检测，并对非 $SZ_l$ 的图输出**所有**不可行的 $\beta$。

本 README 是**主说明文件**：先整体解释两份代码的分工，再**按代码逻辑一步步**详细说明每个脚本的数学建模与实现流程，以便审阅与复现。

> 备注: 运行过程中可能会看到 `libpng warning: iCCP: known incorrect sRGB profile`。这通常是无害的 PNG 颜色配置警告，不会影响图像的保存或显示。

---

## 0. 文件结构与放置建议

推荐将以下文件放在同一目录（相对路径保持不变）：

- `szl_solver.py`
- `generate_nonisomorphic_szl.py`
- `README_CN.md`（本文件）
- `README.md`（英文版）

`generate_nonisomorphic_szl.py` 会动态导入同目录下的 `szl_solver.py` 来获取 `SZlSolver` 类，因此两者需要处于同一目录（或自行修改导入路径）。

运行后将在当前工作目录下创建 `output_{n}v{m}e_l{l}` 文件夹，其中包含总览图 PNG 与结果 TXT 文件。

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

### 2.1 运行枚举 + 总览图 + $SZ_l$ 检测

```bash
python generate_nonisomorphic_szl.py [--vertices n] [--edges m] [--modulus l]
```

默认 `n=5`，`m=21`，`l=6`。示例：`python generate_nonisomorphic_szl.py -n 4 -m 12 -l 5`。

运行后将：

- 在当前工作目录下创建 `output_{n}v{m}e_l{l}` 文件夹
- 在文件夹内保存 PNG 总览图（每张最多 5×5=25 个子图，图多时自动分多张）
- 在控制台打印 $SZ_l$ 判定结果；若某图不是 $SZ_l$，先打印非 $SZ_l$ 图的序号汇总，再逐一输出该图的**所有**不可行 $\beta$（向量形式）
- 将上述运行输出（不含 "Saved: ..." 路径）写入同文件夹下的结果文件 `nonisomorphic_{n}v{m}e_l{l}_results.txt`
- 使用 `--no-log` 可关闭写入结果文件

注意：图不弹窗显示，仅保存到文件（使用 Agg 后端）。

### 2.2 仅运行求解器示例

```bash
python szl_solver.py
```

脚本内置 `main()`，你可以修改其中的 `mod/n/edges/beta` 来测试不同图与不同边界 $\beta$。

---

## 3. `szl_solver.py`：问题定义

### 3.1 合法边界 $\beta$（一般模 $l$）

给定无向连通多重图 $G$（允许重边、不允许自环）与正整数 $l$。

称 $\beta:V(G)\to\mathbb{Z}_{2l}$ 为合法边界（boundary），当且仅当：

- $`\beta(v)\in \{0,1,\dots,2l-1\}`$
- $\beta(v)\equiv \deg(v)\pmod 2$（奇偶性约束）
- $\sum_{v\in V}\beta(v)\equiv 0\pmod{2l}$

奇偶性约束保证了在后续建模中 $\gamma(v)$ 可被 2 整除；后一条件与奇数模类似，因为任意定向都满足 $\sum_v(\text{out}-\text{in})=0$。

### 3.2 $\beta$-定向与 $SZ_l$

对给定 $\beta$，若存在一种对 $G$ 的定向，使得对每个顶点 $v$：

$$
(\mathrm{outdeg}(v)-\mathrm{indeg}(v))\equiv \beta(v)\pmod{2l}
$$

则称该定向为一组 $\beta$-定向。

图 $G$ 称为 $SZ_l$，若对**所有**合法 $\beta$ 都存在 $\beta$-定向。

---

## 4. `szl_solver.py`：数学建模（实现对应）

### 4.1 重边打包：用 $y$ 变量代替逐边方向

与奇数模相同：对无序点对 $\{u,v\}$（规范化为 $u<v$）若有 $k$ 条平行边，令变量 $y\in\{0,1,\dots,k\}$ 表示其中有 $y$ 条边定向为 $u\to v$。则：

- $u$ 点的净贡献（out-in）为 $2y-k$
- $v$ 点的净贡献为 $-(2y-k)$

### 4.2 符号约定与常数项 $C_v$

与奇数模相同：

$$
C_v+2S(v)\equiv \beta(v)\pmod{2l},\qquad
C_v=\sum_{e\ni v}\mathrm{sign}(v,e)\cdot(-k_e)
$$

### 4.3 一般模的关键：除以 2 得到 $\gamma$

当 $l$ 为偶数时，2 在 $\mathbb{Z}_{2l}$ 中不可逆。通过奇偶性约束 $\beta(v)\equiv \deg(v)\pmod 2$，可证 $(\beta(v)-C_v)$ 为偶数（mod $2l$）。定义：

$$
\gamma(v)=\frac{(\beta(v)-C_v)\bmod 2l}{2}\in\mathbb{Z}_l
$$

则约束化为：

$$
S(v)\equiv \gamma(v)\pmod l
$$

代码中把右端预计算为 `target_residue[v_idx]`，后续 DFS 的目标就是让每个顶点的 $S(v)$ 落在对应同余类。

---

## 5. `szl_solver.py`：实现流程（逐步展开）

### 5.1 核心数据结构

- `EdgeBundle(u, v, k)`：记录一个无序点对及重数 $k$（并保证 `u < v`）。
- `OrientationSolution`：保存一组解，包括 `y_by_pair`、`out_minus_in`、`directions` 等（校验模 $2l$）。

### 5.2 `SZlSolver.__init__`：预处理（逐步）

1. **输入校验**：`modulus > 0`；图无自环；图连通（用 `nx.Graph(multigraph)` 去重后检查 `nx.is_connected`）。
2. **顶点与索引**：`vertices = sorted(self.Gm.nodes())`，`index_of_vertex[v] = i`（按顶点标签排序）。
3. **`_collect_edge_bundles()`**：
   - 遍历 `self.Gm.edges()`，对每条边 `(u,v)` 规范化得到 `(a,b)` 其中 `a,b = (u,v) if u<v else (v,u)`；
   - 用字典 `count[(a,b)]` 累加重数；
   - 返回 `[EdgeBundle(u=a, v=b, k=k) for (a,b), k in sorted(count.items())]`，保证点对按字典序排列。
4. **`_build_signs()`**：
   - 对每个边束 `e=(u,v)`（已保证 u<v），设 `sign(u,e)=+1`，`sign(v,e)=-1`；
   - 为每个顶点 `v_idx` 维护列表 `sign_by_vertex[v_idx] = [(eidx, +1 或 -1), ...]`，表示该顶点关联的边束及符号。
5. **`_compute_degrees()`**：对每个边束的 `k`，累加到 `deg[u]` 和 `deg[v]`。
6. **`C_vec`**：对顶点 `v_idx`，`C_vec[v_idx] = sum(sign * (-k_e) for eidx, sign in sign_by_vertex[v_idx])`，即 $C_v$ 的预计算。

### 5.3 `enumerate_betas()`：枚举所有合法 $\beta$（逐步）

1. 对每个顶点 `v`，其合法取值为 $\{x : x\equiv \deg(v)\pmod 2,\; x\in [0,2l)\}$，即 `choices_per_v[i] = [x for x in range(mod2l) if x % 2 == deg[v] % 2]`。
2. 用 `itertools.product(*[choices_per_v[i] for i in range(n-1)])` 枚举前 $n-1$ 个顶点的取值。
3. 对每组 `values`，计算 `s = sum(values) % mod2l`，令 `last = (-s) % mod2l` 使总和为 0（mod $2l$）。
4. 检查 `last % 2 == deg[vertices[-1]] % 2`；若不等则跳过（此时无合法 last）。
5. 构造 `beta = {v: val for v, val in zip(vertices[:-1], values)}`，并设 `beta[vertices[-1]] = last`，`yield beta`。

### 5.4 `solve_for_beta(beta)`：DFS + 剪枝（逐步）

1. **合法性检查**：
   - 顶点集合匹配；
   - 对每个 `v`，`0 <= beta[v] < 2l` 且 `beta[v] % 2 == deg[v] % 2`；
   - `sum(beta.values()) % mod2l == 0`。
2. **计算 target_residue（即 $\gamma$）**：
   - 对每个顶点 `v`，`delta = (beta[v] - C_vec[idx]) % mod2l`；
   - 若 `delta % 2 != 0`，直接返回不可行（无法整除 2）；
   - `target_residue[idx] = (delta // 2) % l`。
3. **建立变量域**：`domains[eidx] = range(0, k_e + 1)`，即 $y_e \in \{0,\dots,k_e\}$。
4. **变量排序**：`order = sorted(range(len(edge_bundles)), key=lambda eidx: len(domains[eidx]))`，域小的边束优先赋值（便于剪枝）。
5. **DFS 状态**：`partial_sum[v_idx]` 表示当前已赋值边束对 $S(v)$ 的整数和；`y_sol[eidx]` 记录解。
6. **`remaining_range_for_vertex(v_idx, next_pos)`**：
   - 对 `order[next_pos:]` 中的边束，若顶点为 `u` 端则贡献 `+y ∈ [0,k]`，为 `v` 端则贡献 `-y ∈ [-k,0]`；
   - 累加得到剩余可达区间 `[L, U]`。
7. **`residue_is_feasible(v_idx, next_pos)`**：
   - `need = (target_residue[v_idx] - partial_sum[v_idx]) % l`；
   - 需存在 $t\in [L,U]$ 使 $t\equiv \text{need}\pmod l$，等价于存在整数 $q$ 使 $L\le \text{need}+ql\le U$；
   - `qmin = ceil_div(L - need, l)`，`qmax = (U - need) // l`；
   - 返回 `qmin <= qmax`。
8. **DFS 递归**：
   - 若 `pos == len(order)`，检查所有顶点 `partial_sum[v_idx] % l == target_residue[v_idx]`；
   - 否则对当前边束 `eidx` 尝试每个 `y in domains[eidx]`，更新 `partial_sum`，若所有顶点 `residue_is_feasible` 且递归成功则记录 `y_sol[eidx]=y` 并返回；否则回滚。
9. **组装解**：由 `y_sol` 计算 `y_by_pair`、`out_minus_in`、`directions`（前 $y$ 条为 $u\to v$，后 $k-y$ 条为 $v\to u$），最后校验 `out_minus_in[v] % mod2l == beta[v]`。

### 5.5 `is_SZl` 与 `get_all_infeasible_betas`

`is_SZl`：枚举所有合法 $\beta$，若发现不可行则返回 `(False, beta)`；否则返回 `(True, None)`。`get_all_infeasible_betas`：收集所有不可行 $\beta$ 的列表并返回。

---

## 6. `generate_nonisomorphic_szl.py`：总体功能

该脚本对**给定的 $n$、$m$、$l$**（均可通过命令行参数指定）执行：

1. 枚举满足约束的 $n$ 顶点、$m$ 条边（计重）多重图：每点最小度 $\ge l-1$，每对顶点边重数 $\le l-2$，边连通度 $\ge 2$，连通
2. 排除含“已知 $SZ_l$”子图的图：3 顶点（边数 $\ge 2l-2$）、4 顶点（当 $n>4$，按论文定理）
3. 在顶点置换意义下去掉同构重复（保留代表元）
4. 绘制所有代表元的总览图（每张最多 5×5 个图，多张时自动分包），保存到输出文件夹
5. 对每个代表元调用 `SZlSolver(..., l)` 判定是否为 $SZ_l$；若非 $SZ_l$，先汇总序号，再逐一打印该图的**所有**不可行 $\beta$

---

## 7. `generate_nonisomorphic_szl.py`：实现流程（逐步展开）

### 7.1 枚举空间与递归生成

- **`make_pairs(n)`**：返回 `[(i,j) for i in 1..n for j in i+1..n]`，即所有无序点对，按字典序。
- **`_enumerate_multiplicities_rec(pairs, remaining, max_per, prefix)`**：
  - 若 `pairs` 为空，则当 `remaining==0` 时 `yield tuple(prefix)`，否则返回；
  - 否则对 `k` 从 `min(remaining, max_per)` 递减到 0，递归调用 `_enumerate_multiplicities_rec(pairs[1:], remaining-k, max_per, prefix+[k])`；
  - 这样逐位枚举，得到所有满足 `sum=k` 且每项 $\le$ max_per 的元组。
- **`enumerate_nonisomorphic_graphs`**：
  1. 调用 `_enumerate_multiplicities_rec(pairs, m, l-2, [])` 得到所有满足总边数 $m$、每对 $\le l-2$ 的元组；
  2. 对每个 `counts`：检查最小度、连通性、边连通度 $\ge 2$、无 3 顶点禁子图、无 4 顶点禁子图（当 $n>4$）；
  3. 用 `canonical_key(counts)` 去重，`reps[key] = counts`，最后返回 `list(reps.values())`。

### 7.2 子图排除（逐步）

- **`has_forbidden_3vertex_subgraph`**：
  - 用 `pair_to_count` 将 `counts` 转为 `(u,v)->w` 的字典；
  - 三重循环枚举所有 3 顶点 `(i,j,k)`，计算 `m_ij + m_ik + m_jk`，若 $\ge 2l-2$ 返回 True。
- **`has_forbidden_4vertex_subgraph`**：
  - 对每个 4 顶点子集 `quad`，取出 6 对的重数并做 SZ_l-simplification（`min(w, l-1)`）；
  - 调用 `_is_4vertex_szl_by_theorem(mult_4, l_val)` 判定该 4 顶点图是否 SZ_l；
  - 若存在则返回 True。
- **`_is_4vertex_szl_by_theorem`**：
  - 检查 $\delta\ge l-1$、$e\ge 3l-3$；
  - 检查 (1) $\mu=l-1$，或 (2) 存在 2 顶点割 $\ne 2l-2$，或 (3) 存在顶点度数 $\not\equiv l\pmod 2$；
  - 任一满足则返回 True。

### 7.3 同构去重：canonical key（逐步）

- **`canonical_key(counts, pairs, n)`**：
  - 先将 `counts` 转为 `w[(u,v)] = 重数` 的字典；
  - 枚举所有 $n!$ 个置换 `perm`，对每个置换构造逆映射 `inv`（新标签->旧标签）；
  - 按固定顺序 `order_pairs = [(1,2),(1,3),...,(n-1,n)]`，对每对 `(a,b)` 用 `inv` 得到原图中的对应点对 `(i_old, j_old)`，取 `w[key]` 得到重数；
  - 得到重标号后的元组 `tup_t`，取字典序最小者作为 `best`；
  - 返回 `best`，用 `reps.setdefault(best, counts)` 去重。

### 7.4 绘图与输出（逐步）

- **`draw_case(ax, counts, pairs, n, ...)`**：
  - 用 `nx.Graph` 构建简单图（仅 presence），采用径向布局：中心 `hub`，其余顶点均匀分布在圆周上；
  - `symmetric_rads(m, step)` 为 $m$ 条平行边生成弯曲半径列表（奇数含 0，偶数不含 0）；
  - 对每对 `(u,v)` 的 `mult` 条边，用 `FancyArrowPatch` 的 `arc3,rad=r` 分别绘制；
  - 固定 `xlim/ylim` 保证视觉尺度一致。
- **主流程中的图片输出**：
  - 每张图最多 25 个子图（5×5）；若 `len(reps) > 25`，则分多张 (`_overview_1.png`, `_overview_2.png`, ...)；
  - 对每张图，`fig, axes = plt.subplots(fig_rows, cols, ...)`，逐个子图调用 `draw_case`，然后 `fig.savefig`、`plt.close`。

### 7.5 批量 $SZ_l$ 判定与输出（逐步）

1. 对每个代表元 `counts`，用 `build_multigraph(counts, pairs, n)` 构造 `nx.MultiGraph`；
2. `solver = SZlSolver(Gm, l_val)`，`infeasible_betas = solver.get_all_infeasible_betas(verbose=False)`；
3. 若有不可行 $\beta$，加入 `non_sz_cases`；
4. 先输出 `Graphs that are NOT SZ_l: #i1, #i2, ...`；
5. 再对每个非 SZ_l 图逐一输出：边重数、度数、所有不可行 $\beta$ 的向量及总数。

---

## 8. 性能与可复现性提示

- **求解器复杂度**：合法 $\beta$ 的数量约 $l^{n-1}$（受奇偶性约束）；单个 $\beta$ 的 DFS 最坏指数级。本实现主要面向中小规模实例的实验验证与复现。
- **确定性**：顶点与边束排序固定；同构去重采用全置换 canonical key，完全确定。
- **与奇数模对比**：奇数模版本可专门针对奇数 $l$ 使用，系数 2 可逆，实现略简；一般模版本统一处理奇偶 $l$，适用于需要偶数的情形。




