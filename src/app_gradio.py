"""
Gradio 前端界面入口
提供 Web 可视化交互，支持问题输入、答案展示、来源查看
启动方式: 在项目根目录执行 python src/app_gradio.py
"""

import os
import sys
import time
import gradio as gr

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.graph.workflow import get_compiled_workflow
from src.graph.state_schema import create_initial_state


# 自定义 CSS 主题
CUSTOM_CSS = """
/* 全局样式 */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

/* 标题样式 */
#main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5em !important;
    font-weight: bold !important;
    margin-bottom: 0.5em !important;
}

#subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1em;
    margin-bottom: 2em;
}

/* 输入框样式 */
#query-input textarea {
    font-size: 1.1em !important;
    padding: 15px !important;
    border-radius: 10px !important;
    border: 2px solid #e0e0e0 !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}

#query-input textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 15px rgba(102, 126, 234, 0.3) !important;
}

/* 提交按钮样式 */
#submit-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 1.1em !important;
    padding: 12px 30px !important;
    border-radius: 10px !important;
    border: none !important;
    cursor: pointer !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

#submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4) !important;
}

/* 清空按钮样式 */
#clear-btn {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    color: white !important;
    font-weight: bold !important;
    padding: 12px 25px !important;
    border-radius: 10px !important;
    border: none !important;
}

#clear-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 20px rgba(245, 87, 108, 0.4) !important;
}

/* 答案展示区 */
#answer-output textarea {
    font-size: 1.1em !important;
    line-height: 1.8 !important;
    padding: 20px !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
}

/* 来源展示区 */
#sources-output textarea {
    font-size: 1em !important;
    padding: 15px !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #fff1eb 0%, #ace0f9 100%) !important;
}

/* 状态信息 */
#status-output textarea {
    font-size: 0.95em !important;
    padding: 10px 15px !important;
    border-radius: 8px !important;
    background: #f8f9fa !important;
}

/* 信息卡片 */
.info-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    border-left: 4px solid #667eea;
}

/* 标签样式 */
.source-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85em;
    margin: 3px;
    font-weight: 500;
}

.source-tag.sql {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    color: #2d3748;
}

.source-tag.doc {
    background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
    color: #2d3748;
}

.source-tag.kg {
    background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
    color: #2d3748;
}

/* 历史记录 */
#history-output textarea {
    font-size: 0.95em !important;
    line-height: 1.6 !important;
    background: #fafafa !important;
}

/* 示例按钮 */
.example-btn {
    background: white !important;
    border: 2px solid #e0e0e0 !important;
    border-radius: 8px !important;
    padding: 10px 15px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    font-size: 0.9em !important;
}

.example-btn:hover {
    border-color: #667eea !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%) !important;
    transform: translateY(-1px) !important;
}

/* 响应时间 */
.response-time {
    text-align: center;
    color: #888;
    font-size: 0.9em;
    margin-top: 10px;
}

/* 侧边栏 */
.sidebar-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 15px;
}

/* 帮助文本 */
.help-text {
    color: #666;
    font-size: 0.9em;
    line-height: 1.6;
}
"""


def get_workflow():
    """获取工作流实例"""
    return get_compiled_workflow()


def process_query(query: str, workflow=None):
    """
    处理用户查询，同步返回结果

    Returns:
        (answer, sources, status_info, history_update)
    """
    if not query.strip():
        return "", "", "请输入您的问题", ""

    if workflow is None:
        workflow = get_workflow()

    start_time = time.time()

    try:
        # 创建初始状态
        initial_state = create_initial_state(query)

        # 执行工作流
        result = workflow.invoke(initial_state)

        # 计算响应时间
        response_time = time.time() - start_time

        # 提取结果
        answer = result.get("answer", "未能生成答案")
        sources = result.get("sources", [])
        confidence = result.get("final_confidence", 0.0)
        routed_sources = result.get("routed_sources", [])
        routing_confidence = result.get("routing_confidence", 0.0)
        conflicts = result.get("conflicts", [])
        source_stats = result.get("source_stats", {})

        # 格式化来源列表
        sources_display = format_sources(sources, routed_sources)

        # 格式化状态信息
        status_info = format_status(
            response_time, confidence, routing_confidence,
            routed_sources, source_stats, conflicts
        )

        # 格式化历史记录
        history_update = format_history_entry(
            query, answer[:100], sources, confidence, response_time
        )

        return answer, sources_display, status_info, history_update

    except Exception as e:
        response_time = time.time() - start_time
        error_msg = f"处理查询时发生错误: {str(e)}"
        return "", "", f"错误: {error_msg}\n响应时间: {response_time:.2f}秒", ""


def format_sources(sources: list, routed_sources: list) -> str:
    """格式化来源展示"""
    if not sources and not routed_sources:
        return "暂无来源信息"

    lines = []
    lines.append("=" * 50)
    lines.append("📚 引用来源")
    lines.append("=" * 50)

    if sources:
        lines.append("\n🏷️  数据来源:")
        for i, source in enumerate(sources, 1):
            lines.append(f"  {i}. {source}")

    if routed_sources:
        lines.append("\n🔀 路由数据源:")
        source_names = {"sql": "SQL数据库", "doc": "文档库", "kg": "知识图谱"}
        for src in routed_sources:
            name = source_names.get(src, src)
            lines.append(f"  • {name}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


def format_status(response_time: float, confidence: float,
                  routing_confidence: float, routed_sources: list,
                  source_stats: dict, conflicts: list) -> str:
    """格式化状态信息"""
    lines = []
    lines.append("=" * 50)
    lines.append("📊 系统状态")
    lines.append("=" * 50)

    # 响应时间
    time_emoji = "⚡" if response_time < 5 else "🔄" if response_time < 10 else "🐌"
    lines.append(f"\n{time_emoji} 响应时间: {response_time:.2f}秒")

    # 置信度
    if confidence >= 0.7:
        conf_emoji = "🟢"
        conf_text = "高置信度"
    elif confidence >= 0.4:
        conf_emoji = "🟡"
        conf_text = "中置信度"
    else:
        conf_emoji = "🔴"
        conf_text = "低置信度"
    lines.append(f"{conf_emoji} 最终置信度: {confidence:.2f} ({conf_text})")

    # 路由信息
    if routed_sources:
        lines.append(f"\n🔀 路由结果: {', '.join(routed_sources)}")
        lines.append(f"   路由置信度: {routing_confidence:.2f}")

    # 数据源统计
    if source_stats:
        stats_parts = []
        for k, v in source_stats.items():
            if v > 0:
                stats_parts.append(f"{k}:{v}")
        if stats_parts:
            lines.append(f"📊 数据源命中: {', '.join(stats_parts)}")

    # 冲突信息
    if conflicts:
        lines.append(f"\n⚠️  检测到 {len(conflicts)} 个冲突")
        for i, conflict in enumerate(conflicts[:3], 1):
            if isinstance(conflict, dict):
                entity = conflict.get("entity", "未知")
                attr = conflict.get("attribute", "未知")
                lines.append(f"   {i}. {entity} - {attr}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


def format_history_entry(query: str, answer_preview: str,
                         sources: list, confidence: float,
                         response_time: float) -> str:
    """格式化历史记录条目"""
    timestamp = time.strftime("%H:%M:%S")
    sources_str = ", ".join(sources) if sources else "无"
    return f"[{timestamp}] {query[:30]}... | 置信度: {confidence:.2f} | 耗时: {response_time:.2f}s | 来源: {sources_str}"


def clear_inputs():
    """清空所有输入和输出"""
    return "", "", "", "", ""


def load_example(example_text: str) -> str:
    """加载示例查询"""
    return example_text


def build_ui():
    """构建 Gradio 界面"""
    # 获取工作流
    workflow = get_workflow()

    # 创建主题
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    )

    with gr.Blocks(
        theme=theme,
        css=CUSTOM_CSS,
        title="多Agent协商式异构数据源融合RAG问答系统",
        analytics_enabled=False
    ) as demo:
        # 标题
        gr.Markdown(
            "# 🎬 多Agent协商式异构数据源融合RAG问答系统",
            elem_id="main-title"
        )
        gr.Markdown(
            "基于多Agent协作的SQL/文档/知识图谱三类异构数据源融合问答",
            elem_id="subtitle"
        )

        with gr.Row():
            # 左侧主区域
            with gr.Column(scale=3):
                # 输入区域
                with gr.Group():
                    query_input = gr.Textbox(
                        label="请输入您的问题",
                        placeholder="例如：阿凡达的剧情介绍、评分最高的5部电影是什么？",
                        lines=3,
                        elem_id="query-input"
                    )

                    with gr.Row():
                        submit_btn = gr.Button(
                            "🚀 发送查询",
                            variant="primary",
                            elem_id="submit-btn",
                            scale=3
                        )
                        clear_btn = gr.Button(
                            "🗑️ 清空",
                            variant="secondary",
                            elem_id="clear-btn",
                            scale=1
                        )

                # 答案展示区
                answer_output = gr.Textbox(
                    label="💡 答案",
                    lines=12,
                    elem_id="answer-output",
                    show_copy_button=True
                )

                # 来源展示区
                sources_output = gr.Textbox(
                    label="📚 来源信息",
                    lines=8,
                    elem_id="sources-output"
                )

            # 右侧信息区
            with gr.Column(scale=1):
                # 状态信息
                status_output = gr.Textbox(
                    label="📊 系统状态",
                    lines=10,
                    elem_id="status-output"
                )

                # 示例查询
                gr.Markdown("### 💡 示例查询")
                examples = [
                    ("SQL查询", "评分最高的5部电影是什么？"),
                    ("文档检索", "阿凡达的剧情介绍"),
                    ("知识图谱", "詹姆斯·卡梅隆导演了哪些电影？"),
                    ("多源融合", "泰坦尼克号的导演还拍过哪些电影？"),
                ]

                for category, example in examples:
                    example_btn = gr.Button(
                        f"[{category}] {example}",
                        variant="secondary",
                        elem_classes="example-btn",
                        size="sm"
                    )
                    example_btn.click(
                        fn=lambda x=example: x,
                        outputs=[query_input]
                    )

                # 帮助信息
                gr.Markdown("""
                ### 📖 使用说明

                1. 在输入框中输入您的问题
                2. 点击「发送查询」按钮
                3. 等待系统处理并返回答案

                **支持的查询类型：**
                - SQL: 电影评分、票房等结构化数据
                - 文档: 电影剧情、幕后信息
                - 知识图谱: 演员、导演关系查询
                - 融合: 综合多个数据源的复杂查询
                """)

                # 历史记录
                history_output = gr.Textbox(
                    label="📝 最近查询记录",
                    lines=6,
                    elem_id="history-output",
                    interactive=False
                )

        # 事件绑定：先清空上次输出（避免旧结果与 loading 动画叠加），再执行查询
        submit_btn.click(
            fn=lambda: ("", "", "[处理中] 正在处理您的问题，请稍候...", ""),
            outputs=[answer_output, sources_output, status_output, history_output],
            queue=False
        ).then(
            fn=lambda q: process_query(q, workflow),
            inputs=[query_input],
            outputs=[answer_output, sources_output, status_output, history_output]
        )

        query_input.submit(
            fn=lambda: ("", "", "[处理中] 正在处理您的问题，请稍候...", ""),
            outputs=[answer_output, sources_output, status_output, history_output],
            queue=False
        ).then(
            fn=lambda q: process_query(q, workflow),
            inputs=[query_input],
            outputs=[answer_output, sources_output, status_output, history_output]
        )

        clear_btn.click(
            fn=clear_inputs,
            outputs=[query_input, answer_output, sources_output, status_output, history_output]
        )

    return demo


def main():
    """主函数"""
    print("=" * 60)
    print("  多Agent协商式异构数据源融合RAG问答系统 - Gradio界面")
    print("=" * 60)
    print("\n正在初始化工作流...")

    # 构建界面
    demo = build_ui()

    print("界面构建完成！")
    print("\n启动Gradio服务...")
    print("访问地址: http://localhost:7860")
    print("=" * 60)

    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
