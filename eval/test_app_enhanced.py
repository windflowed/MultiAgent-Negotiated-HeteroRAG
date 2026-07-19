"""验证增强版app.py功能"""
import sys
sys.path.insert(0, '.')

from src.app import (
    print_welcome_message, print_help, print_history,
    print_stats, print_examples, format_answer_output,
    CLIState, LoadingIndicator
)
import time

# 初始化CLI状态
cli_state = CLIState()

# 测试1：欢迎信息
print("=" * 60)
print("测试1：欢迎信息")
print("=" * 60)
print_welcome_message()

# 测试2：帮助信息
print("\n" + "=" * 60)
print("测试2：帮助信息")
print("=" * 60)
print_help()

# 测试3：示例查询
print("\n" + "=" * 60)
print("测试3：示例查询")
print("=" * 60)
print_examples()

# 测试4：空历史记录
print("\n" + "=" * 60)
print("测试4：空历史记录")
print("=" * 60)
print_history()

# 测试5：添加历史记录后显示
print("\n" + "=" * 60)
print("测试5：历史记录")
print("=" * 60)
cli_state.history.append({
    "query": "评分最高的5部电影是什么？",
    "answer": "评分最高的5部电影为...",
    "sources": ["SQL1"],
    "confidence": 0.85,
    "response_time": 3.25,
    "timestamp": "2024-01-01T12:00:00"
})
cli_state.query_count = 1
cli_state.total_response_time = 3.25
print_history()

# 测试6：统计信息
print("\n" + "=" * 60)
print("测试6：统计信息")
print("=" * 60)
print_stats()

# 测试7：格式化输出
print("\n" + "=" * 60)
print("测试7：格式化输出")
print("=" * 60)
test_result = {
    "answer": "评分最高的5部电影为:肖申克的救赎(8.5分)、教父(8.4分)",
    "sources": ["SQL1", "SQL2"],
    "routed_sources": ["sql"],
    "routing_reason": "查询涉及电影评分数据",
    "routing_confidence": 0.8,
    "final_confidence": 0.85,
    "source_stats": {"sql": 5, "doc": 0, "kg": 0},
    "conflicts": [],
    "error": None
}
cli_state.detailed_mode = True
format_answer_output(test_result)

# 测试8：加载动画
print("\n" + "=" * 60)
print("测试8：加载动画")
print("=" * 60)
loading = LoadingIndicator("测试加载")
loading.start()
time.sleep(2)
loading.stop()
print("加载动画测试完成！")

print("\n" + "=" * 60)
print("所有测试通过！")
print("=" * 60)
