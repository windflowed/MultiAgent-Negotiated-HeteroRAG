"""快速验证三个基线系统可独立运行"""
import sys
sys.path.insert(0, '.')

from eval.baseline import BaselineSingleSourceRAG, BaselineSimpleConcatRAG, BaselineNoNegotiationRAG

test_query = "评分最高的5部电影是什么？"

# 测试基线1
print("=" * 60)
print("测试基线1: 单模态RAG")
print("=" * 60)
try:
    b1 = BaselineSingleSourceRAG()
    r1 = b1.run(test_query)
    print(f"答案: {r1['answer'][:200]}")
    print(f"响应时间: {r1['response_time']:.2f}s")
    print(f"置信度: {r1['final_confidence']:.2f}")
    print("[PASS] 基线1 测试通过")
except Exception as e:
    print(f"[FAIL] 基线1 测试失败: {e}")

print()

# 测试基线2
print("=" * 60)
print("测试基线2: 简单拼接多源RAG")
print("=" * 60)
try:
    b2 = BaselineSimpleConcatRAG()
    r2 = b2.run(test_query)
    print(f"答案: {r2['answer'][:200]}")
    print(f"响应时间: {r2['response_time']:.2f}s")
    print(f"置信度: {r2['final_confidence']:.2f}")
    print("[PASS] 基线2 测试通过")
except Exception as e:
    print(f"[FAIL] 基线2 测试失败: {e}")

print()

# 测试基线3
print("=" * 60)
print("测试基线3: 无协商多Agent RAG")
print("=" * 60)
try:
    b3 = BaselineNoNegotiationRAG()
    r3 = b3.run(test_query)
    print(f"答案: {r3['answer'][:200]}")
    print(f"响应时间: {r3['response_time']:.2f}s")
    print(f"置信度: {r3['final_confidence']:.2f}")
    print("[PASS] 基线3 测试通过")
except Exception as e:
    print(f"[FAIL] 基线3 测试失败: {e}")

print()
print("=" * 60)
print("所有基线系统验证完成！")
print("=" * 60)
