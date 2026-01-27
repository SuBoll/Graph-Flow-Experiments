# `szl_odd_solver.py` 逐行讲解（中文，重点解释 DFS）

> 目标：把 `szl_odd_solver.py` 里每一段代码“为什么这样写、每一行在做什么”讲清楚。  
> 本文特别详细解释 `solve_for_beta()` 里的 `dfs()`（回溯）与剪枝逻辑。

---

## 0. 总体思路（先给你一张“地图”）

这个文件解决的问题是：在 **\(l\) 为奇数** 时，给定一个无向连通多重图 \(G\)，对任意边界函数  
\(\beta:V\to \mathbb{Z}_l\)（取值 \(0,\dots,l-1\)，且 \(\sum_v \beta(v)\equiv 0 \pmod l\)），是否都能找到一个 **\(\beta\)-定向**：

- 给每条边一个方向
- 对每个顶点 \(v\)，要求
  \[
  (\text{outdeg}(v)-\text{indeg}(v)) \equiv \beta(v) \pmod l
  \]

为了节省算力，代码不会逐条边枚举方向，而是把同一对顶点之间的重边 **“打包”** 成一个变量来搜：

- 若顶点对 \(\{u,v\}\)（约定 \(u<v\)）之间有 \(k\) 条平行边  
  设其中有 \(y\) 条定向为 \(u\to v\)，则剩下 \(k-y\) 条为 \(v\to u\)
- 那么在 \(u\) 处的净贡献（out-in）是：
  \[
  y-(k-y)=2y-k
  \]
  在 \(v\) 处净贡献是其相反数。

因此每个顶点对只需要决定一个整数变量 \(y\in[0,k]\)。搜索空间从 \(2^k\) 变成 \(k+1\)。

---

## 1. 文件逐段讲解（带行号）

下面行号来自你当前工作区 `szl_odd_solver.py`（总 318 行）。

---

## 1.1 导入区（L1-L7）

```python
L1: from __future__ import annotations
```
- 作用：允许在类型注解里用字符串或未来形式引用类名，避免前向引用问题（例如在类里用到尚未定义的类名）。

```python
L3: import itertools
```
- 用于枚举 beta（笛卡尔积 `product`），以及在 main 里你构造边列表时常用。

```python
L4: from dataclasses import dataclass
```
- 用 dataclass 简化“数据容器类”的书写：自动生成 `__init__`、`__repr__` 等。

```python
L5: from typing import Dict, List, Tuple, Optional, Iterable
```
- 类型注解用：让函数签名更清晰。

```python
L7: import networkx as nx
```
- 用来存储/传入多重图 `nx.MultiGraph()`，以及连通性检查等。

---

## 1.2 `EdgeBundle`：把重边打包（L10-L15）

```python
L10: @dataclass(frozen=True)
L11: class EdgeBundle:
L12:     """..."""
L13:     u: int
L14:     v: int
L15:     k: int
```

- `EdgeBundle` 表示一个无向顶点对 \(\{u,v\}\) 的“边束”（bundle）。
- `u < v` 是规范化：保证同一对只出现一次，且键稳定。
- `k` 是这对之间的重数（平行边条数）。
- `frozen=True` 表示对象不可变，便于当键、或避免误改。

---

## 1.3 `OddOrientationSolution`：保存一组解（L18-L55）

这个 dataclass 用来把**找到的定向方案**以结构化方式保存，便于打印/验证。

主要字段（L28-L34）：
- `modulus`：模数 \(l\)
- `vertices`：顶点顺序（排序后的顶点列表）
- `edge_bundles`：所有边束
- `y_by_pair`：对每个 `(u,v)`（u<v）存 `y`（有 y 条边定向 u->v）
- `out_minus_in`：每个顶点的 **整数** out-in（不是取模）
- `beta`：当前求解的 beta
- `directions`：逐条边的方向列表 `(tail, head)`，长度等于总边数（计重）

### `pretty_print()`（L36-L54）

这只是打印工具：
- 打印 beta 向量（按 `vertices` 的顺序）
- 打印每个顶点 out-in 以及 mod l 的值
- 打印每对的 `(u,v), k, y, 2y-k`
- 最后打印前 40 条边的方向

这里的关键数学关系：
```text
contrib = 2*y - k
```
就是上面推导的 out-in 贡献。

---

## 1.4 `SZlOddSolver`：核心求解器（L57 起）

这是主类，负责：
- 预处理图
- 枚举 beta
- 对给定 beta 找到定向（solve）
- 判定是否 SZ_l（对所有 beta 都能 solve）

---

## 1.5 `__init__`：预处理（L71-L97）

### 1) 检查输入合法性（L72-L77）

```python
L72: if modulus <= 0 or modulus % 2 == 0:
L73:     raise ValueError(...)
```
只支持**正奇数**模。

```python
L74-L75: if any(u == v for u, v in multigraph.edges()):
```
禁止自环（self-loop），因为自环对 out-in 的贡献会比较特殊（本求解器没处理）。

```python
L76-L77: if not nx.is_connected(nx.Graph(multigraph)):
```
要求图连通。这里把 MultiGraph 转成简单 Graph 来检查连通性（重边不影响连通性）。

### 2) 保存图与模数（L79-L80）

```python
L79: self.Gm = multigraph
L80: self.l = modulus
```

### 3) 顶点顺序与索引（L82-L83）

```python
L82: self.vertices = sorted(self.Gm.nodes())
L83: self.index_of_vertex = {v: i for i, v in enumerate(self.vertices)}
```
给顶点一个固定顺序，后面用数组存储“每个顶点的量”会更快。

### 4) 把重边打包成 `edge_bundles`（L84）

```python
L84: self.edge_bundles = self._collect_edge_bundles()
```
它会把 MultiGraph 里每条边统计到对应 `(min(u,v), max(u,v))` 的计数上，得到重数 k。

### 5) 构建“每个顶点关联哪些边束，以及符号 +1/-1”（L85）

```python
L85: self.sign_by_vertex = self._build_signs()
```
对每条边束 e=(u,v)（u<v）：
- 在 u 处记号 `+1`
- 在 v 处记号 `-1`

这样后面计算某顶点的 sum(sign * y) 就很方便。

### 6) 计算度（L86）

```python
L86: self.deg = self._compute_degrees()
```
这里的“度”按重数计入：`deg(u)+=k, deg(v)+=k`。

### 7) 计算常量项 C_vec（L88-L93）

这是整个算法里最核心的等价变形之一。

每条边束 e=(u,v) 有重数 k、变量 y：
- u 的贡献： \(2y-k\)
- v 的贡献： \(-(2y-k)\)

对某个顶点 v，把所有 incident 边束加起来，可以写成：
\[
\sum_e \text{sign}(v,e)\cdot(2y_e-k_e)
= \underbrace{\sum_e \text{sign}(v,e)\cdot(-k_e)}_{C_v}
 2\sum_e \text{sign}(v,e)\cdot y_e
\]

代码里：
```python
L90-L93: self.C_vec = [ sum(sign * (-k_e)) ... ]
```
也就是把每个顶点的 \(C_v\) 预先算好。

### 8) 计算 2 在 Z_l 的逆元（L95-L96）

```python
L96: self.inv2 = pow(2, -1, self.l)
```
因为 l 是奇数，所以 gcd(2,l)=1，2 在模 l 下可逆。

这一步非常重要：它能把 “2*sum(...) ≡ ...” 变成 “sum(...) ≡ ...”，从而直接做模 l 的线性同余。

---

## 1.6 `_collect_edge_bundles`（L98-L103）

逐行解释：
- `count`：字典，键是 `(a,b)`（a<b），值是重数
- 遍历 `self.Gm.edges()`：MultiGraph 会把每条平行边都枚举出来，所以计数自然得到重数
- 返回 `EdgeBundle` 的列表（按键排序，保证稳定）

---

## 1.7 `_build_signs`（L105-L113）

- 初始化 `sign_by_vertex`：长度 n 的 list，每个元素是一个 list
- 遍历所有边束 eidx, eb：
  - 找到 u,v 的索引 `u_idx, v_idx`
  - 在 u 的列表里 append `(eidx, +1)`
  - 在 v 的列表里 append `(eidx, -1)`

这样后面可快速得到顶点 v 关联的所有边束及符号。

---

## 1.8 `_compute_degrees`（L115-L120）

- 初始化每个顶点度数为 0
- 对每个边束 (u,v,k)：deg[u]+=k, deg[v]+=k

---

## 1.9 `enumerate_betas`：枚举所有 beta（L124-L134）

目的：枚举所有 \(\beta:V\to\mathbb{Z}_l\) 且 \(\sum\beta\equiv 0\pmod l\)。

关键技巧（省一个维度）：
- 任意给前 n-1 个顶点赋值（每个 0..l-1），共有 \(l^{n-1}\) 种
- 最后一个顶点的 beta 必须等于负和：`(-sum) % l` 才能让总和为 0

对应代码：
```python
values in product(range(mod), repeat=n-1)
last = (-s) % mod
```

---

## 1.10 `solve_for_beta`：给定 beta 求定向（L138-L254）

这是最重要的函数。它做三件事：
1) 验证 beta 合法  
2) 把约束变成 “模 l 的线性同余 + 变量 y 的区间”  
3) 用 DFS 搜 y 的一组整数解，然后组装成逐边定向

下面逐段解释。

---

### A. beta 合法性检查（L140-L146）

```python
L140: if set(beta.keys()) != set(self.vertices):
```
必须对每个顶点给值，且不能多/少顶点。

```python
L142-L144:
if any(beta[v] < 0 or beta[v] >= mod):
```
beta 值必须在 0..l-1 范围内。

```python
L145-L146: if sum(beta.values()) % mod != 0
```
beta 总和必须是 0（mod l），否则不可能存在定向（因为所有顶点 out-in 的和恒等于 0）。

---

### B. 计算 gamma（L148-L151）

这一段把原本的约束：
\[
C_v + 2\sum_e \text{sign}(v,e)\,y_e \equiv \beta(v)\pmod l
\]
两边减 C_v，再乘 inv2，得到：
\[
\sum_e \text{sign}(v,e)\,y_e \equiv \gamma(v)\pmod l
\]
其中：
\[
\gamma(v)=2^{-1}\cdot(\beta(v)-C_v)\pmod l
\]

代码：
```python
gamma.append(inv2 * ((beta[v]-C_vec[idx]) % mod) % mod)
```

现在问题就变成：在区间约束 \(y_e\in[0,k_e]\) 下，解一个模 l 的线性同余系统。

---

### C. 建立变量域 domains 与搜索顺序 order（L153-L155）

```python
domains = [ [0..k_e] for each edge bundle e ]
```
每个变量 y_e 可取 0 到 k_e（整数）。

```python
order = sorted(edge indices, key=len(domains[eidx]))
```
按域大小从小到大搜索（k 小的先定），典型的回溯加速策略（类似 CSP 的 MRV）。

---

### D. 搜索状态 y_sol 与 partial_sum（L157-L158）

```python
y_sol: 每条边束的 y 取值（最终答案）\npartial_sum[v_idx]: 当前已赋值边束对顶点 v 的 sum(sign*y) 的“部分和”（整数）
```

为什么 partial_sum 用整数？
- 因为我们既需要做 mod 检查，也需要做“剩余可达区间”的剪枝（区间需要整数上下界）。

---

## 1.11 `remaining_range_for_vertex`：剪枝用的范围估计（L160-L174）

它回答一个问题：
> 从现在开始（next_pos）到搜索结束，剩余未赋值的边束，最多还能让某顶点 v 的 sum(sign*y) 增加多少/减少多少？

对每条未赋值边束 e=(u,v,k)：
- 若当前顶点就是 u：sign=+1，则 sign*y 的范围是 [0, k]
- 若当前顶点就是 v：sign=-1，则 sign*y 的范围是 [-k, 0]
- 若当前顶点不在这条边束上：范围贡献 0

于是把所有未赋值边束的上下界累加，得到总体区间 [L,U]。

代码关键点：
- `L` 只会被减（负方向），`U` 只会被加（正方向）
- 对每条边束，只可能影响两个顶点之一，所以查 `sign_by_vertex[v_idx]` 里是否包含该 eidx

返回 `(L,U)`。

---

## 1.12 DFS 主体：`dfs(pos)`（L176-L218）【重点】

### 变量含义总览

在进入 DFS 之前，重要变量是：
- `order`: 边束索引顺序，例如 [3, 1, 0, 2, ...]
- `pos`: 现在正在决定 order[pos] 这条边束的 y
- `eidx = order[pos]`: 当前要赋值的边束索引
- `domains[eidx]`: y 的可取列表 0..k
- `partial_sum[v_idx]`: 当前已定边束对顶点 v 的部分和（sum(sign*y)）
- `gamma[v_idx]`: 每个顶点的目标同余类（mod l）
- `remaining_range_for_vertex(v_idx, next_pos)`: 剩余边束在 v 上还能贡献的区间

### 1) 递归结束条件（L177-L182）

```python
if pos == len(order):
    for each vertex v:
        if (partial_sum[v]-gamma[v]) % mod != 0: fail
    success
```

解释：
当所有变量 y 都赋值完毕后，partial_sum[v] 就是完整的 \(\sum \text{sign} \cdot y\)。  
它必须满足：
\[
\sum \text{sign}\cdot y \equiv \gamma(v)\pmod l
\]
所以直接做模检查即可。

### 2) 取当前变量（L184-L188）

```python
eidx = order[pos]
eb = edge_bundles[eidx]
u_idx = index_of_vertex[eb.u]
v_idx = index_of_vertex[eb.v]
```

解释：
当前边束的两个端点是 (u,v)，且 u<v。
在 `sign_by_vertex` 的约定下：
- u 端 sign=+1
- v 端 sign=-1

### 3) 枚举 y（L190）

```python
for y in domains[eidx]:   # y=0..k
```

这一步就是“尝试把这对点之间有 y 条边定向为 u->v”。

### 4) 应用该选择：更新 partial_sum（L191-L193）

```python
partial_sum[u_idx] += y
partial_sum[v_idx] -= y
```

因为这一条边束对 u 的贡献（在 sum(sign*y) 这个层面）是 +y，  
对 v 的贡献是 -y。

注意：这里还没乘 2，也没减 k。因为那些都已经被吸收到 C_vec 和 inv2 的变形里了。

### 5) 剪枝：检查“剩余变量是否还能把每个顶点凑到目标同余类”（L195-L208）

这是 DFS 能跑得快的关键。

先解释数学判定：

- 当前某顶点 vv 的 partial_sum[vv] 已经固定了一部分
- 剩余未赋值变量会额外贡献一个整数 t，且 t 一定落在某个区间 [L,U]
- 我们希望最终满足：
  \[
  partial\_sum[vv] + t \equiv \gamma(vv)\pmod l
  \]
  等价于：
  \[
  t \equiv need \pmod l,\quad need = \gamma(vv) - partial\_sum[vv]\ (\bmod\ l)
  \]

所以问题变成：
> 在整数区间 [L,U] 里，是否存在一个数 t，使得 t ≡ need (mod l)？

这等价于：
> 是否存在整数 q 使得 need + q*l ∈ [L,U]？

移项得到：
\[
L-need \le ql \le U-need
\]

令：
```python
left  = L - need
right = U - need
```
则需要存在整数 q，使得 q*l 落在 [left, right]。

进一步：
```python
qmin = ceil(left / l)
qmax = floor(right / l)
```
若 qmin <= qmax，说明存在 q；否则不存在，剪枝。

代码对应（L199-L206）：
```python
need = (gamma[vv] - (partial_sum[vv] % mod)) % mod
left = L - need
right = U - need
if right-left+1 < mod:
    qmin = (left + mod - 1) // mod
    qmax = right // mod
    if qmin > qmax: pruned=True
```

#### 为什么有 `if right-left+1 < mod`？

这是一个“省计算”的小优化：
- 如果区间长度 ≥ mod，那么在一个完整模周期内必定能撞到任意余数类，所以无需严格检查 q 是否存在，直接认为“可达”。
- 只有当区间很短（< mod）时，才需要严格做 qmin/qmax 判定。

因此代码写成：
> 区间短时严格判断，区间长时默认通过。

#### 剪枝失败就 break（L208）
只要某个顶点 vv 无法满足，就直接剪掉这个 y 的分支。

### 6) 递归到下一层（L210-L212）

```python
if not pruned and dfs(next_pos):
    y_sol[eidx] = y
    return True
```

解释：
- 如果当前选择 y 不被剪枝，继续递归
- 一旦下层成功，就记录 y_sol 并一路返回 True（找到一组可行解就停止）

### 7) 回滚（L214-L216）

若该 y 失败（剪枝或下层失败），要把 partial_sum 恢复到进入循环前的状态：
```python
partial_sum[u_idx] -= y
partial_sum[v_idx] += y
```
这是回溯的基本操作。

### 8) 所有 y 都失败则返回 False（L218）

---

## 1.13 搜索入口与结果组装（L220-L254）

```python
ok = dfs(0)
```
从第 0 个变量开始搜索。

若失败，直接 `(False, None)`。

若成功：
1) 读取每条边束的 y
2) 对每条边束计算贡献 `2*y - k` 还原成 out-in
3) 生成逐边方向列表：
   - 先放 y 条 (u->v)
   - 再放 (k-y) 条 (v->u)
4) 最后做一次安全校验：out_minus_in[v] % l == beta[v]

---

## 1.14 `is_SZl`：判定是否对所有 beta 都可行（L256-L270）

逻辑非常直接：
- 枚举所有 beta（共 \(l^{n-1}\) 个）
- 对每个 beta 调 `solve_for_beta`
- 一旦遇到不可行 beta，返回 False 和反例 beta
- 全部可行则返回 True

`verbose` 每 200 个 beta 打一次进度。

---

## 1.15 `build_graph_from_edges` 与 `main`（L273-L316）

- `build_graph_from_edges(n, edges)`：给定顶点数与边列表构造 `nx.MultiGraph`。\n  如果 edges 中有重复的 (u,v)，MultiGraph 会自动形成重边。\n\n- `main()` 是一个演示入口：你可以修改 `mod/n/edges/beta`。\n\n---
\n## 2. 你最关心的：DFS 到底在“搜什么”？\n\n一句话总结：\n> DFS 在搜每个顶点对的 y（有多少条边定向为 u->v），使得所有顶点的模约束同时成立。\n\n- 变量数 = 边束数（不同顶点对的数量）\n- 每个变量的取值数 = k+1（该对重边条数+1）\n- 剪枝通过“剩余可达区间 + 模同余”判断某个顶点是否还可能凑到目标余数\n\n---
\n## 3. 如果你想更直观：给一个小例子\n\n假设某顶点 vv 的目标是：\n- mod=5\n- gamma[vv]=3\n- 当前 partial_sum[vv]=1\n\n则 need = (3-1) mod 5 = 2。\n也就是说：剩余贡献 t 必须满足 t ≡ 2 (mod 5)。\n\n如果剩余边束在 vv 上最多/最少还能给区间 [L,U]=[-3,4]。\n问：[-3,4] 内是否有 ≡2 (mod5) 的数？\n\n- 2 本身就在区间内，所以可达。\n\n若区间是 [-3,1]：\n- 2 不在区间内\n- 2-5=-3 在区间内，所以也可达。\n\n若区间是 [-2,1]：\n- 2 不在区间\n- 2-5=-3 不在区间\n- 2+5=7 不在区间\n所以不可达，必须剪枝。\n\n代码里 qmin/qmax 正是在做这个判断。\n\n---\n\n## 4. 如果你希望“每一行都标注中文注释的代码版本”\n\n我也可以再额外生成一个 `szl_odd_solver_annotated.py`：\n- 保持逻辑不变\n- 每一行后面加中文注释（尤其是 DFS 部分会非常长）\n\n你要这种“可直接对照源码的注释版”吗？\n*** End Patch"}}]}% 92843assistant to=functions.ApplyPatch$commentary  北京赛车冠军_json to=functions.ApplyPatch  天天购彩票 controlled_code
