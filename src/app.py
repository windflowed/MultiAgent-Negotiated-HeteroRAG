"""
命令行交互入口
支持循环输入问题、输出答案与来源信息
增强版：支持历史记录、详细模式、进度提示
"""

import sys
import os
import time
import itertools
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.graph.workflow import get_compiled_workflow
from src.graph.state_schema import create_initial_state


class CLIState:
    """CLI状态管理"""

    def __init__(self):
        self.detailed_mode = False
        self.history: List[Dict[str, Any]] = []
        self.query_count = 0
        self.total_response_time = 0.0


# 全局CLI状态
cli_state = CLIState()


def print_colored(text: str, color: str = "white"):
    """
    打印彩色文本

    Args:
        text: 要打印的文本
        color: 颜色名称 (red, green, yellow, blue, cyan, white)
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    color_code = colors.get(color, colors["white"])
    reset_code = colors["reset"]
    print(f"{color_code}{text}{reset_code}")


def print_welcome_message():
    """打印欢迎信息"""
    print_colored("=" * 60, "cyan")
    print_colored("  多Agent协商式异构数据源融合RAG问答系统", "cyan")
    print_colored("  MultiAgent-Negotiated-HeteroRAG v1.0", "cyan")
    print_colored("=" * 60, "cyan")

    print("\n" + "=" * 60)
    print_colored("支持的查询类型：", "yellow")
    print("  [SQL]     电影评分、票房、上映时间等结构化数据")
    print("  [文档]    电影剧情、幕后信息等非结构化文本")
    print("  [知识图谱] 演员、导演、类型等关系查询")
    print("  [融合]    综合多个数据源的复杂查询")

    print("\n" + "-" * 60)
    print_colored("快捷命令：", "yellow")
    print("  help      - 显示帮助信息")
    print("  history   - 查看历史查询记录")
    print("  stats     - 查看统计信息")
    print("  detail    - 切换详细模式（显示更多调试信息）")
    print("  clear     - 清屏")
    print("  example   - 显示示例查询")
    print("  exit/quit - 退出系统")
    print_colored("=" * 60, "cyan")


def print_help():
    """打印帮助信息"""
    print("\n" + "=" * 60)
    print_colored("命令列表", "yellow")
    print("-" * 60)
    print("  help      显示此帮助信息")
    print("  history   查看历史查询记录（最近10条）")
    print("  stats     查看系统统计信息")
    print("  detail    切换详细模式（显示路由、融合等中间过程）")
    print("  clear     清空屏幕")
    print("  example   显示示例查询")
    print("  exit      退出系统")
    print("  quit      退出系统")
    print("-" * 60)
    print_colored("使用方法：", "yellow")
    print("  直接输入问题即可查询，无需特殊格式")
    print()
    print_colored("示例查询：", "yellow")
    print("  评分最高的5部电影是什么？")
    print("  阿凡达的剧情介绍")
    print("  詹姆斯·卡梅隆导演了哪些电影？")
    print("  阿凡达和泰坦尼克号有什么共同演员？")
    print("  2010年上映的科幻电影有哪些？")
    print("=" * 60)


def print_examples():
    """打印示例查询"""
    examples = [
        ("SQL查询", "评分最高的5部电影是什么？"),
        ("SQL查询", "票房收入最高的电影是哪部？"),
        ("文档检索", "阿凡达的剧情介绍"),
        ("文档检索", "肖申克的救赎讲述了什么故事？"),
        ("知识图谱", "詹姆斯·卡梅隆导演了哪些电影？"),
        ("知识图谱", "阿凡达有哪些演员？"),
        ("多源融合", "阿凡达电影的详细信息"),
        ("多源融合", "泰坦尼克号的导演还拍过哪些电影？"),
    ]

    print("\n" + "=" * 60)
    print_colored("示例查询", "yellow")
    print("-" * 60)
    for category, query in examples:
        print(f"  [{category}] {query}")
    print("=" * 60)


def print_history():
    """打印历史查询记录"""
    print("\n" + "=" * 60)
    print_colored("历史查询记录（最近10条）", "yellow")
    print("-" * 60)

    if not cli_state.history:
        print("  暂无历史记录")
    else:
        for i, record in enumerate(cli_state.history[-10:], 1):
            query = record["query"][:40] + "..." if len(record["query"]) > 40 else record["query"]
            time_str = record.get("response_time", 0)
            print(f"  {i:2d}. {query}")
            print(f"      时间: {time_str:.2f}s | 来源: {', '.join(record.get('sources', []))}")

    print("=" * 60)


def print_stats():
    """打印统计信息"""
    print("\n" + "=" * 60)
    print_colored("系统统计信息", "yellow")
    print("-" * 60)
    print(f"  查询总数: {cli_state.query_count}")

    if cli_state.query_count > 0:
        avg_time = cli_state.total_response_time / cli_state.query_count
        print(f"  平均响应时间: {avg_time:.2f}s")
        print(f"  总响应时间: {cli_state.total_response_time:.2f}s")

    print(f"  详细模式: {'开启' if cli_state.detailed_mode else '关闭'}")
    print(f"  历史记录数: {len(cli_state.history)}")
    print("=" * 60)


class LoadingIndicator:
    """加载动画指示器"""

    def __init__(self, message: str = "处理中"):
        self.message = message
        self.running = False
        self.thread = None

    def start(self):
        """启动加载动画"""
        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def stop(self):
        """停止加载动画"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.1)

    def _animate(self):
        """动画循环"""
        chars = itertools.cycle(["|", "/", "-", "\\"])
        while self.running:
            sys.stdout.write(f"\r  {self.message} {next(chars)} ")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()


def format_answer_output(result: Dict[str, Any]) -> str:
    """
    格式化输出结果

    Args:
        result: 工作流输出结果

    Returns:
        格式化后的输出字符串
    """
    output_lines = []

    # 答案
    answer = result.get("answer", "未能生成答案")
    output_lines.append("\n" + "=" * 60)
    print_colored("回答：", "green")
    print("-" * 60)
    print(answer)

    # 来源信息
    sources = result.get("sources", [])
    if sources:
        print("\n" + "-" * 60)
        print_colored("引用来源：", "cyan")
        print(f"  {', '.join(sources)}")

    # 详细模式：显示更多信息
    if cli_state.detailed_mode:
        print("\n" + "-" * 60)
        print_colored("详细信息：", "yellow")

        # 路由信息
        routed_sources = result.get("routed_sources", [])
        routing_reason = result.get("routing_reason", "")
        routing_confidence = result.get("routing_confidence", 0.0)
        if routed_sources:
            print(f"  [路由] 激活数据源: {', '.join(routed_sources)}")
            if routing_reason:
                print(f"  [路由] 路由原因: {routing_reason}")
            print(f"  [路由] 路由置信度: {routing_confidence:.2f}")

        # 数据源统计
        source_stats = result.get("source_stats", {})
        if source_stats:
            stats_str = ", ".join([f"{k}:{v}" for k, v in source_stats.items() if v > 0])
            if stats_str:
                print(f"  [融合] 数据源统计: {stats_str}")

        # 冲突信息
        conflicts = result.get("conflicts", [])
        if conflicts:
            print(f"  [融合] 检测到冲突: {len(conflicts)} 个")

    # 置信度
    confidence = result.get("final_confidence", 0.0)
    confidence_color = "green" if confidence >= 0.7 else "yellow" if confidence >= 0.4 else "red"
    print("\n" + "-" * 60)
    print_colored(f"置信度: {confidence:.2f}", confidence_color)

    # 错误信息
    error = result.get("error")
    if error:
        print("\n" + "-" * 60)
        print_colored(f"警告: {error}", "red")

    print("=" * 60)

    return ""


def run_interactive_mode():
    """
    运行交互式问答模式
    """
    # 清屏并显示欢迎信息
    os.system("cls" if os.name == "nt" else "clear")
    print_welcome_message()

    # 初始化工作流
    print("\n")
    loading = LoadingIndicator("正在初始化系统")
    loading.start()

    try:
        workflow = get_compiled_workflow()
        loading.stop()
        print_colored("系统初始化成功！", "green")
        print("\n输入您的问题开始查询，输入 help 查看帮助。")
    except Exception as e:
        loading.stop()
        print_colored(f"系统初始化失败: {e}", "red")
        print("请检查配置和数据是否完整。")
        return

    # 交互循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\n[Query] ").strip()

            # 检查空输入
            if not user_input:
                continue

            # 检查退出命令
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n" + "=" * 60)
                print_colored("感谢使用，再见！", "green")
                print("=" * 60)
                break

            # 检查帮助命令
            if user_input.lower() in ["help", "h", "?"]:
                print_help()
                continue

            # 检查历史命令
            if user_input.lower() in ["history", "hist"]:
                print_history()
                continue

            # 检查统计命令
            if user_input.lower() in ["stats", "stat"]:
                print_stats()
                continue

            # 检查详细模式切换
            if user_input.lower() in ["detail", "debug"]:
                cli_state.detailed_mode = not cli_state.detailed_mode
                status = "开启" if cli_state.detailed_mode else "关闭"
                print_colored(f"详细模式已{status}", "yellow")
                continue

            # 检查清屏命令
            if user_input.lower() in ["clear", "cls"]:
                os.system("cls" if os.name == "nt" else "clear")
                print_welcome_message()
                continue

            # 检查示例命令
            if user_input.lower() in ["example", "examples"]:
                print_examples()
                continue

            # 处理查询
            loading = LoadingIndicator("正在处理您的问题")
            loading.start()
            start_time = time.time()

            try:
                # 创建初始状态
                initial_state = create_initial_state(user_input)

                # 执行工作流
                result = workflow.invoke(initial_state)

                # 计算响应时间
                response_time = time.time() - start_time
                loading.stop()

                # 更新统计信息
                cli_state.query_count += 1
                cli_state.total_response_time += response_time

                # 记录历史
                cli_state.history.append({
                    "query": user_input,
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "confidence": result.get("final_confidence", 0.0),
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat()
                })

                # 输出结果
                format_answer_output(result)
                print(f"\n  响应时间: {response_time:.2f}秒")

            except Exception as e:
                loading.stop()
                print_colored(f"\n处理查询时发生错误: {e}", "red")
                print("请重试或输入其他问题。")

        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print_colored("程序被中断，正在退出...", "yellow")
            print("=" * 60)
            break
        except EOFError:
            print("\n\n" + "=" * 60)
            print_colored("输入结束，正在退出...", "yellow")
            print("=" * 60)
            break


def main():
    """主函数"""
    run_interactive_mode()


if __name__ == "__main__":
    main()
