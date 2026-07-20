"""
TASK-603：对比实验脚本
运行主系统 + 3个基线系统，收集对比数据
每类评测场景取1个代表性问题，共5题，4系统对比
"""
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.baseline import (
    BaselineSingleSourceRAG,
    BaselineSimpleConcatRAG,
    BaselineNoNegotiationRAG,
    load_test_dataset
)

from src.graph.state_schema import create_initial_state
from src.graph.workflow import get_compiled_workflow

TEST_QUERIES = {
    "sql_only": "评分最高的5部电影分别是哪些?它们的分数是多少?",
    "doc_only": "阿凡达电影的剧情简介",
    "kg_only": "詹姆斯·卡梅隆导演了哪些电影?",
    "sql_doc": "评分8.0以上且有维基百科词条的电影有哪些?",
    "all_sources": "阿凡达电影的完整信息，包括评分、导演、演员和剧情",
}

print("=" * 70)
print("TASK-603 对比实验：主系统 vs 3个基线")
print("=" * 70)

compiled_workflow = get_compiled_workflow()
print("主系统工作流已编译\n")

all_results = {cat: {} for cat in TEST_QUERIES}

for category, query in TEST_QUERIES.items():
    print("-" * 70)
    print(f"类别: {category} | 查询: {query[:50]}...")
    print("-" * 70)

    # 1. 主系统
    state = create_initial_state(query)
    t0 = time.time()
    try:
        result = compiled_workflow.invoke(state)
        elapsed = time.time() - t0
        all_results[category]["main"] = {
            "answer": result.get("answer", "")[:200],
            "time": round(elapsed, 1),
            "confidence": round(result.get("final_confidence", 0), 2),
            "sources": result.get("sources", []),
            "routed": result.get("routed_sources", []),
            "error": result.get("error")
        }
        print(f"  主系统: {elapsed:.1f}s, confidence={result.get('final_confidence', 0):.2f}, "
              f"routed={result.get('routed_sources', [])}")
    except Exception as e:
        all_results[category]["main"] = {"error": str(e), "time": 0, "answer": ""}
        print(f"  主系统: FAILED ({e})")

    # 2. 基线1：单模态RAG
    try:
        b1 = BaselineSingleSourceRAG()
        t0 = time.time()
        r1 = b1.run(query)
        elapsed = time.time() - t0
        all_results[category]["baseline1"] = {
            "answer": r1.get("answer", "")[:200],
            "time": round(elapsed, 1),
            "confidence": round(r1.get("final_confidence", 0), 2),
            "error": r1.get("error")
        }
        print(f"  基线1(单模态): {elapsed:.1f}s, confidence={r1.get('final_confidence', 0):.2f}")
    except Exception as e:
        all_results[category]["baseline1"] = {"error": str(e), "time": 0, "answer": ""}
        print(f"  基线1(单模态): FAILED ({e})")

    # 3. 基线2：简单拼接
    try:
        b2 = BaselineSimpleConcatRAG()
        t0 = time.time()
        r2 = b2.run(query)
        elapsed = time.time() - t0
        all_results[category]["baseline2"] = {
            "answer": r2.get("answer", "")[:200],
            "time": round(elapsed, 1),
            "confidence": round(r2.get("final_confidence", 0), 2),
            "error": r2.get("error")
        }
        print(f"  基线2(简单拼接): {elapsed:.1f}s, confidence={r2.get('final_confidence', 0):.2f}")
    except Exception as e:
        all_results[category]["baseline2"] = {"error": str(e), "time": 0, "answer": ""}
        print(f"  基线2(简单拼接): FAILED ({e})")

    # 4. 基线3：无协商多Agent
    try:
        b3 = BaselineNoNegotiationRAG()
        t0 = time.time()
        r3 = b3.run(query)
        elapsed = time.time() - t0
        all_results[category]["baseline3"] = {
            "answer": r3.get("answer", "")[:200],
            "time": round(elapsed, 1),
            "confidence": round(r3.get("final_confidence", 0), 2),
            "error": r3.get("error")
        }
        print(f"  基线3(无协商): {elapsed:.1f}s, confidence={r3.get('final_confidence', 0):.2f}")
    except Exception as e:
        all_results[category]["baseline3"] = {"error": str(e), "time": 0, "answer": ""}
        print(f"  基线3(无协商): FAILED ({e})")

# 保存结果
with open("eval/comparison_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print("实验结果已保存到 eval/comparison_results.json")
print("=" * 70)

# 打印汇总对比表
print("\n对比实验结果汇总：\n")
print(f"{'类别':<16} {'主系统':>10} {'基线1(单模态)':>14} {'基线2(拼接)':>13} {'基线3(无协商)':>14}")
print("-" * 72)
for cat, data in all_results.items():
    main_t = data.get("main", {}).get("time", 0)
    b1_t = data.get("baseline1", {}).get("time", 0)
    b2_t = data.get("baseline2", {}).get("time", 0)
    b3_t = data.get("baseline3", {}).get("time", 0)
    print(f"{cat:<16} {f'{main_t:.1f}s':>10} {f'{b1_t:.1f}s':>14} {f'{b2_t:.1f}s':>13} {f'{b3_t:.1f}s':>14}")
    
# 计算平均
avg_main = sum(data.get("main", {}).get("time", 0) for data in all_results.values()) / len(all_results)
avg_b1 = sum(data.get("baseline1", {}).get("time", 0) for data in all_results.values()) / len(all_results)
avg_b2 = sum(data.get("baseline2", {}).get("time", 0) for data in all_results.values()) / len(all_results)
avg_b3 = sum(data.get("baseline3", {}).get("time", 0) for data in all_results.values()) / len(all_results)
print("-" * 72)
print(f"{'平均':<16} {f'{avg_main:.1f}s':>10} {f'{avg_b1:.1f}s':>14} {f'{avg_b2:.1f}s':>13} {f'{avg_b3:.1f}s':>14}")