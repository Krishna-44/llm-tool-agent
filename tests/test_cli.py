import json

from toolagent.cli import main


def test_demo_runs_offline(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Unknown time zone 'Asia/Tokio'" in out
    assert "Total revenue in data/orders.csv is 214.72 USD" in out


def test_tools_command_prints_strict_definitions(capsys, tmp_path):
    assert main(["tools", "--workspace", str(tmp_path)]) == 0
    defs = json.loads(capsys.readouterr().out)
    names = [d["name"] for d in defs]
    assert names == sorted(names) and "write_file" in names
    assert all(d["strict"] and d["input_schema"]["additionalProperties"] is False for d in defs)
