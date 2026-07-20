"""KG Agent 实体消歧修复 - 严格验收测试"""
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.kg_agent import create_kg_agent

print("=" * 60)
print("KG 实体消歧修复 - 严格验收测试")
print("=" * 60)

agent = create_kg_agent()

# ===== 测试用例定义 =====
test_cases = [
    # (query, expected_success, description, validation_func)
    (
        "阿凡达有哪些演员？",
        True,
        "中文电影名单跳查询（movie类型直接使用中文名）",
        lambda r: len(r["triples"]) > 0
    ),
    (
        "詹姆斯·卡梅隆导演了哪些电影？",
        True,
        "中文导演名单跳查询（核心消歧场景：詹姆斯·卡梅隆 → James Cameron）",
        lambda r: (
            len(r["triples"]) > 0 and
            any(e.get("name_en") == "James Cameron" for e in r["entities_found"] if e.get("type") == "director")
        )
    ),
    (
        "莱昂纳多·迪卡普里奥出演了哪些电影？",
        True,
        "中文演员名单跳查询（莱昂纳多·迪卡普里奥 → Leonardo DiCaprio）",
        lambda r: (
            len(r["triples"]) > 0 and
            any(e.get("name_en") == "Leonardo DiCaprio" for e in r["entities_found"] if e.get("type") == "actor")
        )
    ),
    (
        "阿凡达和泰坦尼克号有什么共同演员？",
        True,
        "电影共同邻居查询（两个中文电影名）",
        lambda r: len(r["triples"]) > 0
    ),
    (
        "终结者是什么类型的电影？",
        True,
        "电影类型查询（movie → genre）",
        lambda r: len(r["triples"]) > 0
    ),
]

passed = 0
failed = 0

for query, expected_success, desc, validate in test_cases:
    print(f"\n{'-'*60}")
    print(f"测试: {desc}")
    print(f"查询: {query}")
    
    result = agent.query(query)
    
    # 验证
    try:
        assert result["success"] == expected_success, f"success 期望 {expected_success}，实际 {result['success']}"
        assert validate(result), f"验证函数未通过"
        
        print(f"  成功: {result['success']}")
        print(f"  实体: {[{'name': e['name'], 'name_en': e.get('name_en', 'N/A')} for e in result['entities_found']]}")
        print(f"  三元组数: {len(result['triples'])}")
        for i, t in enumerate(result['triples'][:3]):
            print(f"    {i+1}. {t}")
        print(f"  [PASS]")
        passed += 1
        
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        print(f"  结果: {result}")
        failed += 1
    except Exception as e:
        print(f"  [FAIL] 异常: {e}")
        failed += 1

# ===== 额外验证：实体消歧逻辑 =====
print(f"\n{'='*60}")
print("额外验证：实体消歧逻辑")
print("=" * 60)

# 验证 _disambiguate_entities 方法
test_entities = [
    {"name": "詹姆斯·卡梅隆", "name_en": "James Cameron", "type": "director"},
    {"name": "莱昂纳多·迪卡普里奥", "name_en": "Leonardo DiCaprio", "type": "actor"},
    {"name": "不存在的人", "name_en": "Unknown Person", "type": "actor"},
    {"name": "阿凡达", "name_en": "Avatar", "type": "movie"},
    {"name": "泰坦尼克号", "name_en": "Titanic", "type": "movie"},
]

disambiguated = agent._disambiguate_entities(test_entities)

print("消歧测试:")
all_ok = True

# Director: James Cameron
cameron = next((e for e in disambiguated if e["name"] == "詹姆斯·卡梅隆"), None)
if cameron and cameron.get("name_en") == "James Cameron":
    print(f"  ✓ 詹姆斯·卡梅隆 -> James Cameron (director)")
else:
    print(f"  ✗ 詹姆斯·卡梅隆 消歧失败: {cameron}")
    all_ok = False

# Actor: Leonardo DiCaprio
leo = next((e for e in disambiguated if e["name"] == "莱昂纳多·迪卡普里奥"), None)
if leo and leo.get("name_en") == "Leonardo DiCaprio":
    print(f"  ✓ 莱昂纳多·迪卡普里奥 -> Leonardo DiCaprio (actor)")
else:
    print(f"  ✗ 莱昂纳多·迪卡普里奥 消歧失败: {leo}")
    all_ok = False

# Unknown person: should preserve original
unknown = next((e for e in disambiguated if e["name"] == "不存在的人"), None)
if unknown and unknown.get("name_en") == "Unknown Person":
    print(f"  ✓ 不存在的人 -> Unknown Person (保留原始)")
else:
    print(f"  ✗ 不存在的人 处理异常: {unknown}")
    all_ok = False

# Movie: Avatar -> should use Chinese name
avatar = next((e for e in disambiguated if e["name"] == "阿凡达"), None)
if avatar and avatar.get("name_en") == "阿凡达":
    print(f"  ✓ 阿凡达 -> 阿凡达 (movie使用中文名)")
else:
    print(f"  ✗ 阿凡达 处理异常: {avatar}")
    all_ok = False

# Movie: Titanic -> should use Chinese name
titanic = next((e for e in disambiguated if e["name"] == "泰坦尼克号"), None)
if titanic and titanic.get("name_en") == "泰坦尼克号":
    print(f"  ✓ 泰坦尼克号 -> 泰坦尼克号 (movie使用中文名)")
else:
    print(f"  ✗ 泰坦尼克号 处理异常: {titanic}")
    all_ok = False

if all_ok:
    print("\n[PASS] 实体消歧逻辑验证通过")
else:
    print("\n[FAIL] 实体消歧逻辑验证失败")

# ===== 总结 =====
print(f"\n{'='*60}")
print("验收总结")
print("=" * 60)
print(f"功能测试: {passed}/{passed + failed}")
print(f"逻辑验证: {'PASS' if all_ok else 'FAIL'}")
if failed == 0 and all_ok:
    print("\n✅ 所有验收测试通过！修复成功。")
else:
    print(f"\n❌ 有 {failed} 个测试失败")
print("=" * 60)
