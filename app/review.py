"""盲审分配求解器：OR-Tools CP-SAT 联合分配。

在已完成且结果可行的组卷任务之上，把审阅人一次性联合分配到各卷：
  - 每卷恰好 ``reviewers_per_paper`` 名审阅人，同卷审阅人互不重复
    （0/1 变量天然保证同一人对同一卷至多审一次）；
  - 回避：审阅人不得分配到其声明的不可审卷号；
  - 容量：每名审阅人分到的卷数不超过其容量（跨卷合计）。

择优规则（字典序，逐层最优后固定再优化下一层，与组卷求解器同一风格）：
  1. 最大个人负载（单人分到的卷数）最小；
  2. 各人负载相对平均值的绝对偏差总和最小
     （平均负载 = 总分配数 / 审阅人数，总分配数固定，故为定值）；
  3. 按卷号展平的审阅人编号序列最小：逐卷把“选中审阅人的规范排序编号
     列表”按字典序最小化——等价于把 0/1 选择向量按编号升序逐位取 1 优先
     （即最大化降权编码），分块处理避免指数权重溢出，同时唯一破缺
     审阅人之间的对称性，结果可复现。

每个择优阶段（含编号序列破平的每个分块）都必须取得 OPTIMAL 最优性证明：
任一阶段在限时内只找到可行解而未证明最优、或完全无法判定时返回 error
且**不附任何分配方案**——偏差阶段超时不得退回上一阶段方案（其偏差未必
最小，公开承诺的择优结果会失真），破平分块未完成时此前的解也只对已固定
分块最优。只有首阶段被**证明无解**（INFEASIBLE）时才返回 infeasible 及
各卷可用人力与总容量缺口诊断，绝不返回部分方案。
全部求解单线程、固定随机种子，同一任务同一请求结果可复现。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from .solver import _id_sort_key

CHUNK_BITS = 20  # 编号序列分块编码的块大小（与组卷求解器一致）
STAGE_TIME_LIMIT = 10.0  # 每个择优阶段的求解上限（秒）


def _chunk_expr(
    xp: List[cp_model.IntVar], chunk: int, n: int
) -> cp_model.LinearExpr:
    """第 chunk 块的降权编码：块内小编号占高位（编号越小优先级越高）。

    最大化该编码等价于把“选中编号列表”按字典序最小化：两个等长选择集合
    的字典序比较，取决于对称差中最小的编号属于哪一方。
    """
    lo = chunk * CHUNK_BITS
    hi = min(lo + CHUNK_BITS, n)
    width = hi - lo
    return sum(xp[lo + q] * (1 << (width - 1 - q)) for q in range(width))


def _build_model(
    caps: List[int], excluded: List[set], paper_count: int, per_paper: int
) -> Tuple[
    cp_model.CpModel,
    List[List[cp_model.IntVar]],
    List[cp_model.LinearExpr],
    cp_model.IntVar,
]:
    """联合分配模型：x[r][p] 表示审阅人 r 是否审第 p 卷（全部卷同一模型）。"""
    R = len(caps)
    m = cp_model.CpModel()
    x = [
        [m.NewBoolVar(f"x_{r}_{p}") for p in range(paper_count)]
        for r in range(R)
    ]
    # 每卷恰好 per_paper 名审阅人；0/1 变量保证同卷同一人至多一次
    for p in range(paper_count):
        m.Add(sum(x[r][p] for r in range(R)) == per_paper)
    for r in range(R):
        # 总容量：跨卷合计不超过 capacity
        m.Add(sum(x[r][p] for p in range(paper_count)) <= caps[r])
        # 回避：不可审卷号一律不分配
        for p in excluded[r]:
            m.Add(x[r][p] == 0)
    loads = [sum(x[r][p] for p in range(paper_count)) for r in range(R)]
    max_load = m.NewIntVar(0, paper_count, "max_load")
    for r in range(R):
        m.Add(max_load >= loads[r])
    return m, x, loads, max_load


def _deviation_expr(
    m: cp_model.CpModel, loads: List[cp_model.LinearExpr], total: int
) -> cp_model.LinearExpr:
    """各人负载相对平均值的绝对偏差总和（放大 R 倍整数化）。

    平均负载 = total / R 为定值；Σ|load_r - total/R| 与 Σ|R·load_r - total|
    同序，后者为整数线性表达式，可直接作为 CP-SAT 目标。
    """
    R = len(loads)
    devs = []
    for r in range(R):
        dev = m.NewIntVar(0, R * total, f"dev_{r}")
        m.Add(dev >= R * loads[r] - total)
        m.Add(dev >= total - R * loads[r])
        devs.append(dev)
    return sum(devs) if devs else 0


def _solver(seed: int) -> cp_model.CpSolver:
    s = cp_model.CpSolver()
    s.parameters.num_search_workers = 1  # 单线程保证确定性
    s.parameters.random_seed = seed
    s.parameters.max_time_in_seconds = STAGE_TIME_LIMIT
    return s


def solve_review_plan(
    reviewers: List[Dict[str, Any]],
    reviewers_per_paper: int,
    paper_count: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """求解盲审分配。

    只有全部择优阶段（含编号序列破平的每个分块）都取得最优性证明时才返回
    可行方案；任一阶段在限时内只找到可行解而未证明最优、或完全无法判定时
    返回 error 且不附任何分配方案；只有首阶段被证明无解时才返回 infeasible
    及人力缺口诊断，绝不返回部分方案。

    reviewers 元素为 {"reviewer_id": ..., "capacity": ..., "excluded_papers": [...]}；
    审阅人编号可为数字或字符串，按与题号同一口径规范化排序后参与求解，
    输入顺序不影响分配结果。
    """
    R = len(reviewers)
    P = paper_count
    k = reviewers_per_paper

    # 审阅人按编号规范化排序（与题号同一口径），后续按下标建模
    order = sorted(range(R), key=lambda i: _id_sort_key(reviewers[i]["reviewer_id"]))
    ids = [reviewers[i]["reviewer_id"] for i in order]
    caps = [int(reviewers[i]["capacity"]) for i in order]
    excluded = [{int(p) for p in reviewers[i]["excluded_papers"]} for i in order]

    total = P * k  # 总分配数（定值：每卷恰好 k 人）

    # 无解诊断（纯计数）：各卷可用人力与总容量缺口
    papers_diag = []
    for p in range(P):
        avail = [ids[r] for r in range(R) if p not in excluded[r]]
        papers_diag.append(
            {
                "paper": p,
                "required": k,
                "available_count": len(avail),
                "available_reviewers": avail,
                "shortage": max(0, k - len(avail)),
            }
        )
    # 合计可承接卷数：每人至多审其未回避的卷，再受容量截断
    capacity_total = sum(min(caps[r], P - len(excluded[r])) for r in range(R))
    capacity_gap = max(0, total - capacity_total)

    base: Dict[str, Any] = {
        "paper_count": P,
        "reviewers_per_paper": k,
        "required_total": total,
    }

    def infeasible() -> Dict[str, Any]:
        return {
            "status": "infeasible",
            "message": "不存在满足全部回避与容量约束的盲审分配方案",
            **base,
            "capacity_total": capacity_total,
            "capacity_gap": capacity_gap,
            "papers": papers_diag,
        }

    def error(message: str) -> Dict[str, Any]:
        # 错误结构只含回显与诊断，绝不携带 assignments/loads 等部分方案
        return {"status": "error", **base, "error": message}

    # 阶段 1：最大个人负载最小。必须取得 OPTIMAL：仅 FEASIBLE 时其最大负载
    # 可能并非全局最小，把它固定下来会让后续阶段在错误的上界内继续“择优”。
    m, x, loads, max_load = _build_model(caps, excluded, P, k)
    m.Minimize(max_load)
    s = _solver(seed)
    status = s.Solve(m)
    if status == cp_model.INFEASIBLE:
        # 证明无解：返回既有缺口诊断，不返回任何部分方案
        return infeasible()
    if status == cp_model.FEASIBLE:
        # 只找到可行解而未证明最优：不得固定该负载上界
        return error("求解器在规定时间内无法证明最大个人负载的最优值")
    if status != cp_model.OPTIMAL:
        return error("求解器在规定时间内无法判定可行性")
    best_max = int(s.Value(max_load))

    # 阶段 2：各人负载相对平均值的绝对偏差总和最小。同样必须证明最优：
    # 超时不得退回阶段 1 的方案——其偏差未必最小，公开承诺的择优结果会失真。
    m, x, loads, max_load = _build_model(caps, excluded, P, k)
    m.Add(max_load == best_max)
    deviation = _deviation_expr(m, loads, total)
    m.Minimize(deviation)
    s = _solver(seed)
    status = s.Solve(m)
    # 阶段 1 已证明模型可行，本阶段只增固定等式：INFEASIBLE 只可能来自
    # 求解器数值异常，按“未能证明”处理，绝不退化为任意可行解。
    if status != cp_model.OPTIMAL:
        return error("求解器在规定时间内无法证明负载绝对偏差总和的最优值")
    best_dev = int(s.Value(deviation))

    # 阶段 3：按卷号展平的审阅人编号序列最小。
    # 逐卷、逐块最大化降权编码（等价于选中编号列表的字典序最小化），
    # 并固定此前各卷各块的最优值——卷间与审阅人间的对称被唯一破缺。
    # 每个分块都必须取得 OPTIMAL：首个未完成分块之前的解只对已固定分块
    # 最优，整段序列不满足字典序规则，因此任一未证明分块都直接报错。
    n_chunks = (R + CHUNK_BITS - 1) // CHUNK_BITS
    fixed: List[Tuple[int, int, int]] = []  # (卷 p, 块 k, 块值)
    chosen: Optional[List[List[bool]]] = None
    for p in range(P):
        for ck in range(n_chunks):
            m, x, loads, max_load = _build_model(caps, excluded, P, k)
            m.Add(max_load == best_max)
            m.Add(_deviation_expr(m, loads, total) == best_dev)
            # x 以审阅人为主下标，按卷取列向量（长度 R）参与分块编码
            cols = [[x[r][q] for r in range(R)] for q in range(P)]
            for fp, fk, fv in fixed:
                m.Add(_chunk_expr(cols[fp], fk, R) == fv)
            enc = _chunk_expr(cols[p], ck, R)
            m.Maximize(enc)
            s = _solver(seed)
            status = s.Solve(m)
            if status != cp_model.OPTIMAL:
                return error(
                    "求解器在规定时间内无法证明审阅人编号序列分块的字典序最优"
                )
            fixed.append((p, ck, int(s.Value(enc))))
            chosen = [[bool(s.Value(x[r][q])) for q in range(P)] for r in range(R)]

    # 内部一致性校验：最终分配必须满足全部硬约束并达到已证明的最优值；
    # 不符说明求解过程异常，宁可报错也不返回伪最优方案。
    assert chosen is not None  # P >= 1 且 n_chunks >= 1，阶段 3 循环必然执行
    final_loads = [sum(1 for p in range(P) if chosen[r][p]) for r in range(R)]
    constraints_ok = (
        all(sum(1 for r in range(R) if chosen[r][p]) == k for p in range(P))
        and all(not chosen[r][p] for r in range(R) for p in excluded[r])
        and all(final_loads[r] <= caps[r] for r in range(R))
    )
    final_dev = sum(abs(R * final_loads[r] - total) for r in range(R))
    if not constraints_ok or max(final_loads) != best_max or final_dev != best_dev:
        return error("内部校验失败：最终分配与已证明的最优值不一致")

    assignments = [
        {"paper": p, "reviewer_ids": [ids[r] for r in range(R) if chosen[r][p]]}
        for p in range(P)
    ]
    loads_out = [
        {
            "reviewer_id": ids[r],
            "capacity": caps[r],
            "load": final_loads[r],
        }
        for r in range(R)
    ]
    return {
        "status": "feasible",
        **base,
        "max_load": max(final_loads),
        "assignments": assignments,
        "loads": loads_out,
    }
