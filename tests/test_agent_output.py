"""provider_structured / _dereference_defs 回归测试。

背景：嵌套 pydantic 模型的 json_schema 带 ``$defs``/``$ref``，dmxapi 路由到 Gemini 原生后端
时被翻译成 ``generation_config.response_schema``（OpenAPI 子集），后者不认 ``$defs`` →
400 "Unknown name $defs"（生产实测 critic 连续 400 重试）。这里锁住「发出的 schema 必须平化」
这一契约，防止未来改 schema 时悄悄把 $ref 带回来。完全离线。
"""

from __future__ import annotations

import json

from img_iter_agent.agents._agent_output import (
    CriticAgentOutput,
    _dereference_defs,
    provider_structured,
)
from img_iter_agent.calibration.creativity_tuner import CreativityRenovation


def test_raw_pydantic_schema_has_defs():
    """前提确认：裸 pydantic schema 确实带 $defs/$ref（即被修复的根因）。"""
    raw = json.dumps(CriticAgentOutput.model_json_schema())
    assert "$defs" in raw and '"$ref"' in raw


def test_provider_structured_flattens_defs():
    """critic 的 schema 平化后无 $defs/$ref，且嵌套结构逐层完整。"""
    js = provider_structured(CriticAgentOutput).schema_spec.json_schema
    txt = json.dumps(js)
    assert "$defs" not in txt
    assert '"$ref"' not in txt
    # 结构没被内联破坏：dimensions[].items[]（逐项判定）仍是完整 object
    dim = js["properties"]["dimensions"]["items"]
    assert dim["type"] == "object"
    item = dim["properties"]["items"]["items"]
    assert item["type"] == "object"
    assert set(item["properties"]) >= {"id", "passed"}


def test_creativity_renovation_flattened():
    """creativity_tuner 的 schema（items 嵌 CreativityRenoItem）同样平化。"""
    js = provider_structured(CreativityRenovation).schema_spec.json_schema
    txt = json.dumps(js)
    assert "$defs" not in txt and '"$ref"' not in txt
    assert js["properties"]["items"]["items"]["type"] == "object"


def test_deref_sibling_keys_override_ref_target():
    """$ref 兄弟键应覆盖被引定义展开后的同名键（JSON Schema 语义）。"""
    schema = {
        "$defs": {"X": {"type": "object", "properties": {"a": {"type": "string"}}}},
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/X", "description": "override"}},
    }
    out = _dereference_defs(schema)
    x = out["properties"]["x"]
    assert x["type"] == "object"
    assert x["description"] == "override"
    assert "a" in x["properties"]


def test_deref_circular_ref_no_infinite_recursion():
    """循环引用不无限递归：环处放弃内联、剥掉 $ref 退化成空 schema（防御路径，本项目无环）。"""
    schema = {
        "$defs": {"A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/A"}}}},
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/A"}},
    }
    out = _dereference_defs(schema)
    txt = json.dumps(out)
    assert "$defs" not in txt and '"$ref"' not in txt
    # a 本体成功内联；环处 b 退化为空 schema（无限深约束被丢弃，换取可终止）
    assert out["properties"]["a"]["type"] == "object"
    assert out["properties"]["a"]["properties"]["b"] == {}


def test_provider_structured_dict_passthrough():
    """已是 dict 的 schema 不做平化（调用方自管，比如已平化的 schema）。"""
    d = {"type": "object", "properties": {"a": {"type": "string"}}}
    s = provider_structured(d)
    assert s.schema_spec.json_schema is d
