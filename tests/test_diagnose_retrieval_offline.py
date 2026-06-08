"""diagnose_retrieval.py --offline 약식 스모크 테스트 (VPN/Qdrant 없이 전체 경로 검증).

in-memory 코퍼스 + fake client로 production multiview 경로가 의도대로 정렬하는지 확인:
- 이상헌(M0001, joint) 최상위(coverage=joint).
- 균형 separate(권영수) > 한쪽몰빵 separate(김편중) — balance가 몰빵 억제.
- 반도체-only(박반도)/AI-only(정평가)는 partial → fallback tier로 강등.
- off-topic(최무관, M0005)은 어떤 view에도 안 잡혀 production 랭킹에서 제외.
"""
from __future__ import annotations

import json
import os
import tempfile

from apps.core import llm_policies
from apps.tools import diagnose_retrieval
from apps.tools.diagnose_retrieval import main


def test_diagnostic_llm_policy_uses_high_reasoning(monkeypatch):
    monkeypatch.setattr(llm_policies, "CONSISTENCY_REASONING_EFFORT", "low")
    monkeypatch.setattr(llm_policies, "CONSISTENCY_INCLUDE_REASONING", True)
    monkeypatch.setattr(llm_policies, "CONSISTENCY_DISABLE_THINKING", True)

    diagnose_retrieval._apply_diagnostic_llm_policy()

    invoke_kwargs = llm_policies.build_consistency_invoke_kwargs()
    assert invoke_kwargs["reasoning_effort"] == "high"
    assert invoke_kwargs["include_reasoning"] is False
    assert invoke_kwargs["disable_thinking"] is False


# registry 제거 후 offline 모드는 LLM이 없어 concept_specs를 못 만든다. production 랭킹(joint/
# balance/fallback) 검증을 위해 LLM 플래너 산출을 --plan-json으로 시뮬레이션한다(과거 registry가
# 감지하던 ai/semiconductor 정의를 그대로 복제 → 결과 동일). 합성/감지가 아니라 planner-source.
_AI_SEMI_PLAN_JSON = json.dumps({"concept_specs": [
    {"id": "ai", "label": "인공지능", "role": "required",
     "query_terms": ["인공지능", "AI", "머신러닝", "딥러닝", "신경망"],
     "evidence_terms": ["인공지능", "AI", "머신러닝", "machine learning", "딥러닝",
                        "deep learning", "신경망", "neural network", "뉴로모픽",
                        "neuromorphic", "LLM", "생성형", "AI반도체", "인공지능반도체", "지능형반도체"],
     "weak_terms": ["지능형", "스마트", "자동화"]},
    {"id": "semiconductor", "label": "반도체", "role": "required",
     "query_terms": ["반도체", "시스템반도체", "반도체소자"],
     "evidence_terms": ["반도체", "시스템반도체", "AI반도체", "인공지능반도체", "지능형반도체",
                        "반도체소자", "반도체설계", "반도체공정", "반도체장비", "반도체소재",
                        "집적회로", "웨이퍼", "파운드리", "SoC", "CMOS", "MOSFET", "DRAM",
                        "SRAM", "PIM", "NPU", "FPGA", "ASIC", "HBM", "칩"],
     "weak_terms": ["지능형", "시스템", "센서", "회로", "소자", "공정", "산업", "개발"]},
]})


def _run_offline(capsys, query: str = "인공지능 반도체") -> str:
    fd, plan_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_AI_SEMI_PLAN_JSON)
        rc = main(["--offline", "--query", query, "--keywords", "인공지능", "반도체",
                   "--plan-json", plan_path])
        assert rc == 0
        return capsys.readouterr().out
    finally:
        os.unlink(plan_path)


def _multiview_block(out: str) -> str:
    start = out.index("[Multiview Flat Relevance")
    end = out.index("[Sparse SPLADE Search]")
    return out[start:end]


def test_offline_runs_end_to_end(capsys):
    out = _run_offline(capsys)
    assert "offline=True" in out
    assert "[Multiview Flat Relevance" in out
    assert "[Legacy Grouped Path" in out  # 진단/AB 경로 보존
    assert "[Post-Group Relevance Gate]" in out


def test_offline_concept_evidence_plan_and_signals(capsys):
    out = _run_offline(capsys)
    # 동적 Concept Evidence Plan 상세(evidence/weak 분리) 출력.
    assert "[Concept Evidence Plan]" in out
    assert "evidence(strong)=" in out
    assert "weak=" in out
    # chunk별 concept 신호(confirmed/weak_only/view_only) 출력.
    assert "signals={" in out
    assert "'confirmed':" in out


def test_offline_plan_json_injection_uses_planner_source(capsys, tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"concept_specs": [
            {"id": "ai", "label": "인공지능", "role": "required",
             "query_terms": ["인공지능"], "evidence_terms": ["인공지능", "딥러닝"], "weak_terms": ["지능형"]},
            {"id": "semiconductor", "label": "반도체", "role": "required",
             "query_terms": ["반도체"], "evidence_terms": ["반도체"], "weak_terms": ["시스템"]},
        ]}),
        encoding="utf-8",
    )
    rc = main(["--offline", "--query", "인공지능 반도체", "--keywords", "인공지능", "반도체",
               "--plan-json", str(plan_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "source=planner" in out          # 주입한 concept_specs 사용
    assert "src=planner" in out             # spec.source = planner


def test_offline_production_ranks_joint_then_balanced_then_onesided(capsys):
    mv = _multiview_block(_run_offline(capsys))
    # 멀티뷰 production 섹션 내 등장 순서 = 최종 정렬.
    pos = {rid: mv.index(rid) for rid in ("M0001", "M0002", "M0003", "M0004", "M0006")}
    assert pos["M0001"] < pos["M0002"] < pos["M0003"]  # joint > 균형 separate > 몰빵 separate
    assert pos["M0003"] < pos["M0004"]  # main(separate) > fallback(partial)
    assert pos["M0003"] < pos["M0006"]


def test_offline_joint_candidate_is_top_main(capsys):
    mv = _multiview_block(_run_offline(capsys))
    first_line = next(line for line in mv.splitlines() if line.strip().startswith("1."))
    assert "[MAIN]" in first_line
    assert "researcher_id=M0001" in first_line
    assert "coverage='joint'" in first_line
    assert "target_researcher_rank=1" in mv


def test_offline_partial_candidates_in_fallback_tier(capsys):
    mv = _multiview_block(_run_offline(capsys))
    fallback_lines = [line for line in mv.splitlines() if "[FALLBACK]" in line]
    fallback_ids = " ".join(fallback_lines)
    assert "M0004" in fallback_ids  # 반도체-only
    assert "M0006" in fallback_ids  # AI-only
    assert all("coverage='partial'" in line for line in fallback_lines)


def test_offline_offtopic_excluded_from_production_ranking(capsys):
    mv = _multiview_block(_run_offline(capsys))
    assert "M0005" not in mv  # off-topic 연구자는 회수 0 → 결과 제외
    assert "최무관" not in mv


def test_offline_generic_query_without_concepts(capsys):
    # 전부 일반어(GENERIC_SEARCH_TERMS) → query_exact 합성도 비어 source=none, gate 비활성, 충돌 없이 실행.
    rc = main(["--offline", "--query", "연구 산업 개발", "--keywords", "연구", "산업", "개발"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"source": "none"' in out
    assert "gate_enabled=False" in out
