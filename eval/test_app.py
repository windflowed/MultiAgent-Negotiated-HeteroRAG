"""验证app.py可独立运行"""
import sys
sys.path.insert(0, '.')

from src.app import format_answer_output, print_welcome_message, print_help

# 测试1：格式化输出
print("=" * 60)
print("测试1：格式化输出")
print("=" * 60)
test_result = {
    "answer": "评分最高的5部电影为:肖申克的救赎(8.5分)、教父(8.4分)、千与千寻(8.3分)",
    "sources": ["SQL1", "SQL2"],
    "routed_sources": ["sql"],
    "routing_reason": "查询涉及电影评分数据",
    "final_confidence": 0.85,
    "error": None
}
formatted = format_answer_output(test_result)
print(formatted)

# 测试2：欢迎信息
print("\n" + "=" * 60)
print("测试2：欢迎信息")
print("=" * 60)
print_welcome_message()

# 测试3：帮助信息
print("\n" + "=" * 60)
print("测试3：帮助信息")
print("=" * 60)
print_help()

# 测试4：带错误的结果
print("\n" + "=" * 60)
print("测试4：带错误的结果")
print("=" * 60)
error_result = {
    "answer": "",
    "sources": [],
    "routed_sources": ["sql"],
    "routing_reason": "路由正常",
    "final_confidence": 0.0,
    "error": "SQL查询执行失败"
}
formatted_error = format_answer_output(error_result)
print(formatted_error)

print("\n" + "=" * 60)
print("所有测试通过！")
print("=" * 60)
