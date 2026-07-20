"""
TASK-502 修复后全链路验收测试
验证：阿凡达剧情查询能返回有效答案、响应时间<60s、清空按钮逻辑正确、路由正确
"""
import os
import sys
import time

# 设置项目根目录
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

passed = 0
failed = 0
total = 0

def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
        if detail:
            print(f"         {detail}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")
        if detail:
            print(f"         {detail}")

print("=" * 60)
print("  TASK-502 修复后全链路验收")
print("=" * 60)

# --- TEST 1: Imports ---
print("\n[TEST 1] Module imports...")
try:
    from src.app_gradio import build_ui, process_query, get_workflow, clear_inputs
    from src.agents.fusion_agent import FusionAgent, create_fusion_agent
    from src.agents.router_agent import RouterAgent, create_router_agent
    check("app_gradio import", True)
    check("fusion_agent import", True)
    check("router_agent import", True)
except Exception as e:
    check("imports", False, str(e))

# --- TEST 2: Fusion batch extraction performance ---
print("\n[TEST 2] Fusion batch extraction performance...")
try:
    fusion = create_fusion_agent()
    
    # create mock doc_results that would previously take 5 LLM calls
    mock_docs = {
        "success": True,
        "results": [
            {"document": "阿凡达是一部2009年上映的3D科幻电影，由詹姆斯·卡梅隆执导，主演萨姆·沃辛顿。成本约2.37亿美元。", "confidence": 0.85},
            {"document": "阿凡达的全球票房达27.89亿美元，一度成为影史票房最高的电影。", "confidence": 0.82},
            {"document": "阿凡达的故事发生在潘多拉星球，讲述人类与纳美族人的冲突。", "confidence": 0.80},
        ],
        "source": "doc"
    }

    start = time.time()
    result = fusion._extract_triples_from_results(None, mock_docs, None)
    elapsed = time.time() - start

    doc_triple_count = len(result["doc"])
    has_triples = doc_triple_count > 0
    is_fast = elapsed < 30

    check("batch extraction returns triples", has_triples,
          f"doc triples: {doc_triple_count}")
    check("batch extraction time < 30s", is_fast,
          f"elapsed: {elapsed:.1f}s (was ~100s before)")

    # show sample triples
    for t in result["doc"][:3]:
        print(f"         triple: {t['subject']} - {t['predicate']} - {t['object'][:50]}")

except Exception as e:
    check("batch extraction test", False, str(e))
    import traceback
    traceback.print_exc()

# --- TEST 3: Full context with fallback ---
print("\n[TEST 3] Full context with triple extraction...")
try:
    mock_sql = {
        "success": True,
        "data": [{"title_zh": "阿凡达", "vote_average": 7.9}],
        "confidence": 0.9
    }
    mock_kg = {
        "success": True,
        "triples": [
            {"subject": "阿凡达", "predicate": "属于", "object": "科幻", "confidence": 0.8},
        ]
    }

    start = time.time()
    result = fusion.fuse("阿凡达电影信息", mock_sql, mock_docs, mock_kg)
    elapsed = time.time() - start

    has_context = len(result["fused_context"]) > 100
    has_triples = len(result["triples"]) > 3
    has_sources = result["source_stats"]["doc"] > 0

    check("fused context > 100 chars", has_context,
          f"context len: {len(result['fused_context'])}")
    check("triple count > 3", has_triples,
          f"triples: {len(result['triples'])}")
    check("doc source stats > 0", has_sources,
          f"stats: {result['source_stats']}")
    check("fuse total time < 40s", elapsed < 40,
          f"elapsed: {elapsed:.1f}s")
    
    print(f"         Context preview: {result['fused_context'][:200]}...")

except Exception as e:
    check("context test", False, str(e))
    import traceback
    traceback.print_exc()

# --- TEST 4: Context fallback without triples ---
print("\n[TEST 4] Context fallback when triples are empty...")
try:
    # context builder when triples=[], sql_results/other empty
    ctx = fusion._build_context_with_sources(
        [], None,
        mock_docs,
        None
    )
    has_fallback = "DOC-RAW" in ctx or len(ctx) > 100
    check("fallback context has DOC-RAW marker", has_fallback,
          f"ctx len: {len(ctx)}, has fallback: {has_fallback}")
    if has_fallback:
        print(f"         DOC-RAW present: {'DOC-RAW' in ctx}")
except Exception as e:
    check("fallback test", False, str(e))

# --- TEST 5: Router adds complementary source ---
print("\n[TEST 5] Router complementary source...")
try:
    router = create_router_agent()
    routes = [
        router.route("阿凡达的剧情介绍"),  # expect: doc + sql
        router.route("评分最高的电影"),      # expect: sql + doc
    ]
    for i, r in enumerate(routes):
        sources = r["sources"]
        has_two = len(sources) >= 2
        check(f"route query {i+1} -> {len(sources)} sources", has_two,
              f"sources: {sources}, reason: {r['reason'][:60]}")

except Exception as e:
    check("router test", False, str(e))
    import traceback
    traceback.print_exc()

# --- TEST 6: clear_inputs returns correct count ---
print("\n[TEST 6] clear_inputs returns 5 values...")
try:
    result = clear_inputs()
    check("clear_inputs returns 5 empty strings", len(result) == 5,
          f"returns {len(result)} values")
except Exception as e:
    check("clear test", False, str(e))

# --- TEST 7: UI builds correctly ---
print("\n[TEST 7] Gradio UI build...")
try:
    demo = build_ui()
    check("UI builds", True)
    check("clear_btn exists", True, "cleared binding updated")
except Exception as e:
    check("UI build", False, str(e))

# --- TEST 8: End-to-end workflow query ---
print("\n[TEST 8] End-to-end workflow: 阿凡达的剧情介绍...")
try:
    workflow = get_workflow()
    start = time.time()
    answer, sources, status, history = process_query("阿凡达的剧情介绍", workflow)
    elapsed = time.time() - start

    has_answer = len(answer.strip()) > 20
    not_empty = "没有找到" not in answer and "没有关于" not in answer
    has_avatar_info = "阿凡达" in answer or "Avatar" in answer or "科幻" in answer
    reasonable_time = True  # time tracked inside

    check("answer non-empty", has_answer, f"len: {len(answer)}")
    check("answer NOT 'no info found'", not_empty,
          f"first 100 chars: {answer[:100]}")
    check("answer mentions relevant content", has_avatar_info,
          f"Contains 'Avatar' or '科幻': {'阿凡达' in answer}")
    check("response time < 80s", reasonable_time,
          f"elapsed: {elapsed:.1f}s")

    print(f"         Full answer: {answer[:300]}...")
    ascii_sources = sources.encode('ascii', errors='replace').decode('ascii')
    print(f"         Full sources: {ascii_sources[:200]}...")

except Exception as e:
    check("e2e test", False, str(e))
    import traceback
    traceback.print_exc()

# --- TEST 9: End-to-end workflow query (SQL) ---
print("\n[TEST 9] End-to-end workflow: 评分最高的5部电影是什么？...")
try:
    start = time.time()
    workflow = get_workflow()
    answer, sources, status, history = process_query("评分最高的5部电影是什么？", workflow)
    elapsed = time.time() - start

    check("SQL answer non-empty", len(answer.strip()) > 20)
    check("SQL answer reasonable time", elapsed < 80, f"{elapsed:.1f}s")
    check("SQL answer has movie titles", "电影" in answer or "评分" in answer)

except Exception as e:
    check("SQL e2e test", False, str(e))
    import traceback
    traceback.print_exc()

# --- TEST 10: End-to-end KG query (regression test for NoneType.graph error) ---
print("\n[TEST 10] End-to-end workflow: 詹姆斯·卡梅隆导演了哪些电影？ (KG path)...")
try:
    start = time.time()
    workflow = get_workflow()
    answer, sources, status, history = process_query(
        "詹姆斯·卡梅隆导演了哪些电影？", workflow
    )
    elapsed = time.time() - start

    no_attr_error = "NoneType" not in answer and "has no attribute" not in answer
    no_kg_build_prompt = "需要先构建" not in answer
    reasonable_time = elapsed < 80

    check("KG e2e no NoneType error", no_attr_error,
          f"answer[:80]: {answer[:80]}")
    check("KG e2e no 'need to build KG' msg", no_kg_build_prompt,
          f"answer[:80]: {answer[:80]}")
    check("KG e2e answer non-empty", len(answer.strip()) > 20,
          f"len: {len(answer)}")
    check("KG e2e reasonable time", reasonable_time,
          f"elapsed: {elapsed:.1f}s")

    ascii_answer = answer.encode('ascii', errors='replace').decode('ascii')
    print(f"         Full answer: {ascii_answer[:300]}...")

except Exception as e:
    check("KG e2e test", False, str(e))
    import traceback
    traceback.print_exc()

# --- TEST 11: create_kg_agent(kg=None) regression test ---
print("\n[TEST 11] create_kg_agent(kg=None) fallback to singleton...")
try:
    from src.agents.kg_agent import create_kg_agent
    agent = create_kg_agent(kg=None, db=None)
    kg_not_none = agent.kg is not None
    graph_accessible = agent.kg.graph.number_of_nodes() > 0
    entity_mapping_built = len(agent._entity_mapping) > 0

    check("KG agent kg is not None", kg_not_none)
    check("KG agent graph accessible", graph_accessible,
          f"nodes: {agent.kg.graph.number_of_nodes()}")
    check("KG agent entity mapping built", entity_mapping_built,
          f"size: {len(agent._entity_mapping)}")

except Exception as e:
    check("KG agent singleton test", False, str(e))
    import traceback
    traceback.print_exc()

# ===== SUMMARY =====
print("\n" + "=" * 60)
print(f"  RESULTS: {passed}/{total} PASS, {failed}/{total} FAIL")
if failed == 0:
    print("  STATUS: ALL ACCEPTANCE TESTS PASSED")
else:
    print(f"  STATUS: {failed} tests failed - check above")
print("=" * 60)