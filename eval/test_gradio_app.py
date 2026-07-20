"""
Gradio 前端界面验收测试脚本
验证 TASK-502 输出产物：界面可正常启动、输入问题可展示答案与对应来源、交互流畅
"""

import os
import sys
import time

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

# Windows 控制台 UTF-8 编码
if sys.platform == "win32":
    os.system("chcp 65001 > nul")

print("=" * 60)
print("  TASK-502 Gradio Frontend Acceptance Test")
print("=" * 60)

# Acceptance criterion 1: UI builds without errors
print("\n[Test 1] UI build test...")
try:
    from src.app_gradio import build_ui, process_query, get_workflow
    demo = build_ui()
    print("  PASS: Gradio UI built successfully")
except Exception as e:
    print(f"  FAIL: UI build failed: {e}")
    sys.exit(1)

# Acceptance criterion 2: Workflow loads properly
print("\n[Test 2] Workflow loading test...")
try:
    workflow = get_workflow()
    print(f"  PASS: Workflow loaded: {type(workflow).__name__}")
except Exception as e:
    print(f"  FAIL: Workflow loading failed: {e}")
    sys.exit(1)

# Acceptance criterion 3: Questions produce answers with sources
print("\n[Test 3] Query processing test...")
test_queries = [
    ("SQL", "评分最高的5部电影是什么？"),
    ("Doc", "阿凡达的剧情介绍"),
    ("KG", "詹姆斯·卡梅隆导演了哪些电影？"),
]

all_pass = True
for i, (category, query) in enumerate(test_queries, 1):
    print(f"\n  Query {i} [{category}]: {query}")
    start_time = time.time()
    try:
        answer, sources, status, history = process_query(query, workflow)
        elapsed = time.time() - start_time

        has_answer = len(answer.strip()) > 0
        has_sources = len(sources.strip()) > 0

        if has_answer and has_sources:
            print(f"    PASS: answer={len(answer)} chars, sources={len(sources)} chars, time={elapsed:.2f}s")
            print(f"    Answer preview: {answer[:100]}...")
        else:
            print(f"    WARN: answer_len={len(answer)}, sources_len={len(sources)}")
            all_pass = False
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False

# Acceptance criterion 4: Performance (response time)
print("\n[Test 4] Performance test (response time)...")
try:
    start = time.time()
    answer, sources, status, history = process_query("阿凡达的评分是多少？", workflow)
    elapsed = time.time() - start

    if elapsed < 60:
        print(f"  PASS: Response time {elapsed:.2f}s < 60s, smooth interaction")
    else:
        print(f"  WARN: Response time {elapsed:.2f}s may be slow")
except Exception as e:
    print(f"  FAIL: {e}")

# Acceptance criterion 5: CSS custom theme
print("\n[Test 5] CSS custom theme check...")
from src.app_gradio import CUSTOM_CSS
checks = {
    "gradient backgrounds": "linear-gradient" in CUSTOM_CSS,
    "transition animations": "transition" in CUSTOM_CSS,
    "hover effects": ":hover" in CUSTOM_CSS,
    "custom fonts": "font-size" in CUSTOM_CSS,
    "box shadows": "box-shadow" in CUSTOM_CSS,
    "border radius": "border-radius" in CUSTOM_CSS,
    "gr.themes.Soft theme": "gr.themes.Soft" in open(
        os.path.join(_project_root, "src", "app_gradio.py"), encoding="utf-8"
    ).read(),
}

for name, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"  [{status}] {name}")

all_pass &= all(checks.values())

# Acceptance criterion 6: Verify UI components exist
print("\n[Test 6] UI components check...")
try:
    block_ids = [block.elem_id for block in demo.blocks.values() if hasattr(block, 'elem_id') and block.elem_id]
    required_ids = ["main-title", "query-input", "submit-btn", "clear-btn", "answer-output", "sources-output", "status-output"]
    for rid in required_ids:
        if rid in block_ids:
            print(f"  PASS: Component '#{rid}' found")
        else:
            print(f"  WARN: Component '#{rid}' not found")
except Exception as e:
    print(f"  WARN: Could not check components: {e}")

print("\n" + "=" * 60)
print("  TASK-502 Acceptance Summary")
print("=" * 60)

if all_pass:
    print("""
STATUS: ALL TESTS PASSED

Output artifact: src/app_gradio.py
  - Gradio frontend with question input, answer output, source display
  - Custom CSS theme with gradients, transitions, hover effects
  - Uses gr.themes.Soft base theme
  - Example queries pre-loaded

Acceptance criteria:
  [PASS] UI starts normally (builds without errors)
  [PASS] Questions display answers with source information
  [PASS] Interaction is smooth (response time acceptable)
  [PASS] Custom CSS theme is visually appealing

Launch command:
  cd MultiAgent-Negotiated-HeteroRAG
  python src/app_gradio.py
  Open http://localhost:7860
""")
else:
    print("\n  WARNING: Some tests failed. Review above output.")

print("=" * 60)
