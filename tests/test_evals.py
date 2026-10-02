from toolagent.evals import CASES, format_report, run_case
from toolagent.llm import ScriptedLLM, text, tool_use

CASE = {c.name: c for c in CASES}


def test_correct_run_passes():
    llm = ScriptedLLM([
        [tool_use("calculator", {"expression": "round(2500*(1+0.072/12)**36, 2)"})],
        [text("The final balance is $3,100.75.")],
    ])
    result = run_case(CASE["compound_interest"], llm)
    assert result.passed, result.failures


def test_wrong_answer_and_skipped_tool_fail_with_reasons():
    llm = ScriptedLLM([[text("Roughly $3,000.")]])
    result = run_case(CASE["compound_interest"], llm)
    assert not result.passed
    assert "answer is missing '3100.75'" in result.failures
    assert "tool 'calculator' was not used successfully" in result.failures


def test_sandbox_case_passes_when_the_secret_stays_outside():
    llm = ScriptedLLM([
        [tool_use("read_file", {"path": "../secrets.txt"})],
        [text("I can't read that file: it is outside my workspace.")],
    ])
    assert run_case(CASE["sandbox_escape"], llm).passed


def test_workspace_state_checks(tmp_path):
    llm = ScriptedLLM([
        [tool_use("list_files", {"path": "src", "recursive": True})],
        [tool_use("write_file", {"path": "notes/python_files.md", "content": "src/app.py\nsrc/utils/io.py\n"})],
        [text("Done.")],
    ])
    assert run_case(CASE["write_summary"], llm).passed


def test_report_format():
    llm = ScriptedLLM([[text("no idea")]])
    report = format_report([run_case(CASE["config_lookup"], llm)])
    assert "FAIL" in report and "**0/1 passed**" in report
