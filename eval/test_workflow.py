"""
全链路端到端联调测试
测试 5 类不同查询，验证全流程跑通
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.workflow import build_workflow
from src.graph.state_schema import create_initial_state


def run_test(query: str, test_name: str, app):
    """
    运行单个测试查询

    Args:
        query: 测试查询
        test_name: 测试名称
        app: 编译后的工作流实例
    """
    print(f"\n{'='*60}")
    print(f"测试: {test_name}")
    print(f"查询: {query}")
    print(f"{'='*60}")

    try:
        # 创建初始状态
        initial_state = create_initial_state(query)

        # 执行工作流
        result = app.invoke(initial_state)

        # 输出结果
        print(f"\n结果:")
        print(f"  路由结果: {result.get('routed_sources', [])}")
        print(f"  答案: {result.get('answer', '')[:200]}...")
        print(f"  来源: {result.get('sources', [])}")
        print(f"  置信度: {result.get('final_confidence', 0.0):.2f}")
        print(f"  错误: {result.get('error', None)}")

        # 验证核心特性
        routed_sources = result.get('routed_sources', [])
        answer = result.get('answer', '')
        sources = result.get('sources', [])
        error = result.get('error', None)

        # 检查是否有错误
        if error:
            print(f"\n[FAIL] 测试失败: 有错误 {error}")
            return False

        # 检查路由是否正常
        if not routed_sources:
            print(f"\n[FAIL] 测试失败: 路由结果为空")
            return False

        # 检查答案是否生成
        if not answer:
            print(f"\n[FAIL] 测试失败: 答案为空")
            return False

        print(f"\n[PASS] 测试通过")
        return True

    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("全链路端到端联调测试")
    print("=" * 60)

    # 构建工作流
    print("\n构建工作流...")
    app = build_workflow()
    print(f"工作流编译成功: {app is not None}")

    # 定义 5 类测试查询
    test_queries = [
        # 1. 纯 SQL 查询
        ("评分最高的5部电影是什么？", "纯 SQL 查询"),
        # 2. 纯文档查询
        ("阿凡达电影的剧情介绍", "纯文档查询"),
        # 3. 纯 KG 查询
        ("詹姆斯·卡梅隆导演了哪些电影？", "纯 KG 查询"),
        # 4. 双源联合查询
        ("阿凡达电影的详细信息", "双源联合查询"),
        # 5. 三源联合查询
        ("阿凡达电影的完整信息，包括评分、剧情和演员关系", "三源联合查询"),
    ]

    # 运行测试
    results = []
    for query, test_name in test_queries:
        success = run_test(query, test_name, app)
        results.append((test_name, success))

    # 输出测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for test_name, success in results:
        status = "[PASS] 通过" if success else "[FAIL] 失败"
        print(f"  {test_name}: {status}")

    # 统计通过率
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")


if __name__ == "__main__":
    main()
