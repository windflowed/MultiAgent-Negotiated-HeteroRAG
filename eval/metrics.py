"""
评价指标计算脚本
实现时延自动统计、批量评测、结果导出功能，配套人工评分模板

核心评价指标：
1. 答案准确率（Answer Accuracy）：生成答案与标准答案的匹配度
2. 信息完整度（Information Completeness）：答案覆盖关键信息点的比例
3. 冲突消解正确率（Conflict Resolution Accuracy）：冲突用例中正确消解的比例
4. 平均响应时延（Average Response Time）：所有查询的平均响应时间
"""

import os
import sys
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class EvaluationResult:
    """单条评测结果"""
    question_id: int
    question: str
    expected_answer: str
    generated_answer: str
    category: str
    difficulty: str
    conflict: bool
    data_sources: List[str]
    response_time: float
    confidence: float
    sources_used: List[str]
    error: Optional[str]
    human_scores: Optional[Dict[str, float]] = None


@dataclass
class SystemMetrics:
    """系统级评测指标"""
    system_name: str
    total_questions: int
    avg_response_time: float
    avg_confidence: float
    accuracy_score: float
    completeness_score: float
    conflict_resolution_score: float
    category_scores: Dict[str, float]
    error_count: int
    success_rate: float


class MetricsCalculator:
    """
    评价指标计算器
    支持批量评测、指标计算、结果导出
    """

    def __init__(self, test_dataset_path: str = "eval/test_dataset.json"):
        """
        初始化评测器

        Args:
            test_dataset_path: 评测数据集路径
        """
        self.test_dataset_path = test_dataset_path
        self.test_dataset = self._load_test_dataset()

    def _load_test_dataset(self) -> List[Dict]:
        """加载评测数据集"""
        with open(self.test_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("questions", [])

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度（简化版：基于关键词匹配）

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数 0.0-1.0
        """
        if not text1 or not text2:
            return 0.0

        # 简单的关键词匹配
        keywords1 = set(text1)
        keywords2 = set(text2)

        intersection = keywords1 & keywords2
        union = keywords1 | keywords2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _calculate_accuracy_score(
        self,
        generated_answer: str,
        expected_answer: str,
        verification: Dict
    ) -> float:
        """
        计算答案准确率

        Args:
            generated_answer: 生成的答案
            expected_answer: 标准答案
            verification: 验证规则

        Returns:
            准确率分数 0.0-1.0
        """
        if not generated_answer:
            return 0.0

        score = 0.0
        total_checks = 0

        # 检查必须包含的关键词
        contains = verification.get("contains", [])
        if contains:
            matched = sum(1 for kw in contains if kw in generated_answer)
            score += matched / len(contains)
            total_checks += 1

        # 检查不能包含的关键词
        not_contains = verification.get("not_contains", [])
        if not_contains:
            not_matched = sum(1 for kw in not_contains if kw not in generated_answer)
            score += not_matched / len(not_contains)
            total_checks += 1

        # 检查数值范围
        numeric_range = verification.get("numeric_range")
        if numeric_range and len(numeric_range) == 2:
            # 尝试从答案中提取数字
            import re
            numbers = re.findall(r'\d+\.?\d*', generated_answer)
            if numbers:
                for num_str in numbers:
                    try:
                        num = float(num_str)
                        if numeric_range[0] <= num <= numeric_range[1]:
                            score += 1.0
                            break
                    except ValueError:
                        continue
            total_checks += 1

        # 检查最少电影数
        min_movies = verification.get("min_movies")
        if min_movies:
            # 简单统计答案中提到的电影数
            import re
            movie_pattern = r'《([^》]+)》|([^\s,，。、]+(?:电影|影片))'
            movies_found = len(re.findall(movie_pattern, generated_answer))
            if movies_found >= min_movies:
                score += 1.0
            total_checks += 1

        return score / total_checks if total_checks > 0 else 0.0

    def _calculate_completeness_score(
        self,
        generated_answer: str,
        expected_answer: str
    ) -> float:
        """
        计算信息完整度

        Args:
            generated_answer: 生成的答案
            expected_answer: 标准答案

        Returns:
            完整度分数 0.0-1.0
        """
        if not expected_answer:
            return 1.0 if generated_answer else 0.0

        # 提取标准答案中的关键信息点
        import re
        # 提取数字
        expected_numbers = set(re.findall(r'\d+\.?\d*', expected_answer))
        generated_numbers = set(re.findall(r'\d+\.?\d*', generated_answer))

        # 提取中文关键词（长度>1的中文词）
        expected_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', expected_answer))
        generated_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', generated_answer))

        # 计算数字覆盖率
        number_coverage = 0.0
        if expected_numbers:
            matched_numbers = expected_numbers & generated_numbers
            number_coverage = len(matched_numbers) / len(expected_numbers)

        # 计算关键词覆盖率
        keyword_coverage = 0.0
        if expected_keywords:
            matched_keywords = expected_keywords & generated_keywords
            keyword_coverage = len(matched_keywords) / len(expected_keywords)

        # 综合完整度（数字权重0.4，关键词权重0.6）
        return 0.4 * number_coverage + 0.6 * keyword_coverage

    def _calculate_conflict_resolution_score(
        self,
        generated_answer: str,
        expected_answer: str,
        is_conflict: bool
    ) -> float:
        """
        计算冲突消解正确率

        Args:
            generated_answer: 生成的答案
            expected_answer: 标准答案
            is_conflict: 是否为冲突用例

        Returns:
            冲突消解分数 0.0-1.0
        """
        if not is_conflict:
            return 1.0  # 非冲突用例默认满分

        # 对于冲突用例，检查答案是否与标准答案一致
        return self._calculate_text_similarity(generated_answer, expected_answer)

    def run_system_evaluation(
        self,
        system_runner: Callable,
        system_name: str,
        max_questions: int = None
    ) -> SystemMetrics:
        """
        运行系统评测

        Args:
            system_runner: 系统运行函数，输入(query, expected_answer)，输出Dict
            system_name: 系统名称
            max_questions: 最大测试问题数

        Returns:
            系统评测指标
        """
        results = []
        questions_to_test = (
            self.test_dataset[:max_questions]
            if max_questions
            else self.test_dataset
        )

        for item in questions_to_test:
            question_id = item.get("id", 0)
            question = item.get("question", "")
            expected_answer = item.get("expected_answer", "")
            category = item.get("category", "")
            difficulty = item.get("difficulty", "")
            conflict = item.get("conflict", False)
            data_sources = item.get("data_sources", [])
            verification = item.get("verification", {})

            # 运行系统
            start_time = time.time()
            try:
                result = system_runner(question, expected_answer)
                response_time = time.time() - start_time

                generated_answer = result.get("answer", "")
                confidence = result.get("confidence", 0.0)
                sources_used = result.get("sources", [])
                error = result.get("error")

                # 计算各项指标
                accuracy = self._calculate_accuracy_score(
                    generated_answer, expected_answer, verification
                )
                completeness = self._calculate_completeness_score(
                    generated_answer, expected_answer
                )
                conflict_resolution = self._calculate_conflict_resolution_score(
                    generated_answer, expected_answer, conflict
                )

                results.append({
                    "question_id": question_id,
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": generated_answer,
                    "category": category,
                    "difficulty": difficulty,
                    "conflict": conflict,
                    "data_sources": data_sources,
                    "response_time": response_time,
                    "confidence": confidence,
                    "sources_used": sources_used,
                    "error": error,
                    "accuracy": accuracy,
                    "completeness": completeness,
                    "conflict_resolution": conflict_resolution
                })

            except Exception as e:
                results.append({
                    "question_id": question_id,
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": "",
                    "category": category,
                    "difficulty": difficulty,
                    "conflict": conflict,
                    "data_sources": data_sources,
                    "response_time": 0.0,
                    "confidence": 0.0,
                    "sources_used": [],
                    "error": str(e),
                    "accuracy": 0.0,
                    "completeness": 0.0,
                    "conflict_resolution": 0.0
                })

        # 计算系统级指标
        return self._aggregate_metrics(results, system_name)

    def _aggregate_metrics(
        self,
        results: List[Dict],
        system_name: str
    ) -> SystemMetrics:
        """
        聚合评测结果为系统级指标

        Args:
            results: 评测结果列表
            system_name: 系统名称

        Returns:
            系统评测指标
        """
        if not results:
            return SystemMetrics(
                system_name=system_name,
                total_questions=0,
                avg_response_time=0.0,
                avg_confidence=0.0,
                accuracy_score=0.0,
                completeness_score=0.0,
                conflict_resolution_score=0.0,
                category_scores={},
                error_count=0,
                success_rate=0.0
            )

        total = len(results)
        error_count = sum(1 for r in results if r.get("error"))
        success_rate = (total - error_count) / total

        # 平均响应时间
        avg_response_time = sum(r["response_time"] for r in results) / total

        # 平均置信度
        avg_confidence = sum(r["confidence"] for r in results) / total

        # 平均准确率
        accuracy_score = sum(r["accuracy"] for r in results) / total

        # 平均完整度
        completeness_score = sum(r["completeness"] for r in results) / total

        # 冲突消解正确率（仅统计冲突用例）
        conflict_results = [r for r in results if r["conflict"]]
        if conflict_results:
            conflict_resolution_score = (
                sum(r["conflict_resolution"] for r in conflict_results)
                / len(conflict_results)
            )
        else:
            conflict_resolution_score = 1.0

        # 分类别准确率
        categories = set(r["category"] for r in results)
        category_scores = {}
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            if cat_results:
                category_scores[cat] = (
                    sum(r["accuracy"] for r in cat_results) / len(cat_results)
                )

        return SystemMetrics(
            system_name=system_name,
            total_questions=total,
            avg_response_time=avg_response_time,
            avg_confidence=avg_confidence,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            conflict_resolution_score=conflict_resolution_score,
            category_scores=category_scores,
            error_count=error_count,
            success_rate=success_rate
        )

    def export_results_json(
        self,
        results: List[Dict],
        output_path: str
    ):
        """
        导出评测结果为JSON格式

        Args:
            results: 评测结果列表
            output_path: 输出文件路径
        """
        output_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_questions": len(results),
                "results": results
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"评测结果已导出到: {output_path}")

    def export_results_csv(
        self,
        results: List[Dict],
        output_path: str
    ):
        """
        导出评测结果为CSV格式

        Args:
            results: 评测结果列表
            output_path: 输出文件路径
        """
        if not results:
            print("无评测结果可导出")
            return

        # CSV列定义
        fieldnames = [
            "question_id", "question", "expected_answer", "generated_answer",
            "category", "difficulty", "conflict", "data_sources",
            "response_time", "confidence", "sources_used", "error",
            "accuracy", "completeness", "conflict_resolution",
            "human_accuracy", "human_completeness", "human_notes"
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                # 创建行数据，添加人工评分占位列
                row = {
                    "question_id": result.get("question_id"),
                    "question": result.get("question"),
                    "expected_answer": result.get("expected_answer"),
                    "generated_answer": result.get("generated_answer"),
                    "category": result.get("category"),
                    "difficulty": result.get("difficulty"),
                    "conflict": result.get("conflict"),
                    "data_sources": ",".join(result.get("data_sources", [])),
                    "response_time": f"{result.get('response_time', 0):.2f}",
                    "confidence": f"{result.get('confidence', 0):.2f}",
                    "sources_used": ",".join(result.get("sources_used", [])),
                    "error": result.get("error", ""),
                    "accuracy": f"{result.get('accuracy', 0):.2f}",
                    "completeness": f"{result.get('completeness', 0):.2f}",
                    "conflict_resolution": f"{result.get('conflict_resolution', 0):.2f}",
                    "human_accuracy": "",
                    "human_completeness": "",
                    "human_notes": ""
                }
                writer.writerow(row)

        print(f"评测结果已导出到: {output_path}")

    def generate_human_scoring_template(
        self,
        results: List[Dict],
        output_path: str
    ):
        """
        生成人工评分模板

        Args:
            results: 评测结果列表
            output_path: 输出文件路径
        """
        template_data = {
            "metadata": {
                "description": "人工评分模板",
                "instructions": [
                    "1. 请在'human_accuracy'列填写准确率评分（0-1）",
                    "2. 请在'human_completeness'列填写完整度评分（0-1）",
                    "3. 请在'human_notes'列填写评审意见",
                    "4. 评分标准：0=完全错误，0.5=部分正确，1.0=完全正确"
                ],
                "created_at": datetime.now().isoformat()
            },
            "questions": []
        }

        for result in results:
            template_data["questions"].append({
                "question_id": result.get("question_id"),
                "question": result.get("question"),
                "expected_answer": result.get("expected_answer"),
                "generated_answer": result.get("generated_answer"),
                "system_accuracy": result.get("accuracy", 0),
                "system_completeness": result.get("completeness", 0),
                "human_accuracy": None,
                "human_completeness": None,
                "human_notes": ""
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)

        print(f"人工评分模板已生成: {output_path}")

    def print_summary(self, metrics: SystemMetrics):
        """
        打印评测摘要

        Args:
            metrics: 系统评测指标
        """
        print("\n" + "=" * 60)
        print(f"评测摘要: {metrics.system_name}")
        print("=" * 60)
        print(f"总问题数: {metrics.total_questions}")
        print(f"成功运行率: {metrics.success_rate:.2%}")
        print(f"错误数: {metrics.error_count}")
        print("\n--- 核心指标 ---")
        print(f"平均响应时延: {metrics.avg_response_time:.2f}s")
        print(f"平均置信度: {metrics.avg_confidence:.2f}")
        print(f"答案准确率: {metrics.accuracy_score:.2%}")
        print(f"信息完整度: {metrics.completeness_score:.2%}")
        print(f"冲突消解正确率: {metrics.conflict_resolution_score:.2%}")
        print("\n--- 分类别准确率 ---")
        for category, score in metrics.category_scores.items():
            print(f"  {category}: {score:.2%}")
        print("=" * 60)


def create_main_system_runner():
    """
    创建主系统运行器

    Returns:
        系统运行函数
    """
    from src.graph.workflow import get_compiled_workflow
    from src.graph.state_schema import create_initial_state

    workflow = get_compiled_workflow()

    def runner(query: str, expected_answer: str) -> Dict[str, Any]:
        """
        运行主系统

        Args:
            query: 用户查询
            expected_answer: 标准答案（未使用）

        Returns:
            系统输出结果
        """
        state = create_initial_state(query)
        result = workflow.invoke(state)

        return {
            "answer": result.get("answer", ""),
            "confidence": result.get("final_confidence", 0.0),
            "sources": result.get("sources", []),
            "error": result.get("error")
        }

    return runner


def create_baseline_runner(baseline_class):
    """
    创建基线系统运行器

    Args:
        baseline_class: 基线系统类

    Returns:
        系统运行函数
    """
    def runner(query: str, expected_answer: str) -> Dict[str, Any]:
        """
        运行基线系统

        Args:
            query: 用户查询
            expected_answer: 标准答案（未使用）

        Returns:
            系统输出结果
        """
        baseline = baseline_class()
        result = baseline.run(query)

        return {
            "answer": result.get("answer", ""),
            "confidence": result.get("final_confidence", 0.0),
            "sources": result.get("sources", []),
            "error": result.get("error")
        }

    return runner


if __name__ == "__main__":
    print("=" * 60)
    print("评测指标计算脚本")
    print("=" * 60)

    # 初始化评测器
    calculator = MetricsCalculator()

    # 示例：生成人工评分模板
    print("\n生成人工评分模板...")
    sample_results = [
        {
            "question_id": 1,
            "question": "数据库中评分最高的5部电影分别是哪些?",
            "expected_answer": "评分最高的5部电影为:肖申克的救赎(8.5分)、教父(8.4分)",
            "generated_answer": "评分最高的电影是肖申克的救赎(8.5分)",
            "accuracy": 0.8,
            "completeness": 0.6
        }
    ]
    calculator.generate_human_scoring_template(
        sample_results,
        "eval/human_scoring_template.json"
    )

    print("\n" + "=" * 60)
    print("评测脚本使用说明")
    print("=" * 60)
    print("""
使用方法:
1. 评测主系统:
   from eval.metrics import MetricsCalculator, create_main_system_runner
   calculator = MetricsCalculator()
   runner = create_main_system_runner()
   metrics = calculator.run_system_evaluation(runner, "主系统", max_questions=5)
   calculator.print_summary(metrics)

2. 评测基线系统:
   from eval.baseline import BaselineSingleSourceRAG
   from eval.metrics import MetricsCalculator, create_baseline_runner
   calculator = MetricsCalculator()
   runner = create_baseline_runner(BaselineSingleSourceRAG)
   metrics = calculator.run_system_evaluation(runner, "基线1-单模态RAG")
   calculator.print_summary(metrics)

3. 导出结果:
   calculator.export_results_json(results, "eval/results.json")
   calculator.export_results_csv(results, "eval/results.csv")
    """)
