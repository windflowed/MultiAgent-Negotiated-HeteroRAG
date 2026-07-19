"""验证metrics.py可独立运行"""
import sys
sys.path.insert(0, '.')

from eval.metrics import MetricsCalculator

# 测试1：加载评测数据集
print("=" * 60)
print("测试1：加载评测数据集")
print("=" * 60)
calc = MetricsCalculator()
print(f"加载成功，共 {len(calc.test_dataset)} 个问题")

# 测试2：查看第一个问题
print("\n" + "=" * 60)
print("测试2：查看第一个问题")
print("=" * 60)
q = calc.test_dataset[0]
print(f"ID: {q['id']}")
print(f"问题: {q['question']}")
print(f"类别: {q['category']}")
print(f"难度: {q['difficulty']}")

# 测试3：计算准确率
print("\n" + "=" * 60)
print("测试3：计算准确率")
print("=" * 60)
verification = q.get('verification', {})
accuracy = calc._calculate_accuracy_score(
    "评分最高的5部电影为:肖申克的救赎(8.5分)、教父(8.4分)、千与千寻(8.3分)",
    q['expected_answer'],
    verification
)
print(f"准确率分数: {accuracy:.2f}")

# 测试4：计算完整度
print("\n" + "=" * 60)
print("测试4：计算完整度")
print("=" * 60)
completeness = calc._calculate_completeness_score(
    "评分最高的5部电影为:肖申克的救赎(8.5分)、教父(8.4分)、千与千寻(8.3分)",
    q['expected_answer']
)
print(f"完整度分数: {completeness:.2f}")

# 测试5：生成人工评分模板
print("\n" + "=" * 60)
print("测试5：生成人工评分模板")
print("=" * 60)
sample_results = [{
    "question_id": 1,
    "question": q['question'],
    "expected_answer": q['expected_answer'],
    "generated_answer": "评分最高的5部电影为:肖申克的救赎(8.5分)、教父(8.4分)",
    "accuracy": 0.8,
    "completeness": 0.6
}]
calc.generate_human_scoring_template(sample_results, "eval/human_scoring_template.json")
print("人工评分模板生成成功")

# 测试6：导出CSV
print("\n" + "=" * 60)
print("测试6：导出CSV")
print("=" * 60)
calc.export_results_csv(sample_results, "eval/test_results.csv")
print("CSV导出成功")

print("\n" + "=" * 60)
print("所有测试通过！")
print("=" * 60)
