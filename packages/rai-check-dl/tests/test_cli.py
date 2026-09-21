from pathlib import Path

import typer
from rai_audit.dl.cli import register
from typer.testing import CliRunner


def test_cli_writes_medical_json_report(tmp_path: Path):
    data = tmp_path / "predictions.csv"
    data.write_text(
        "y_true,y_pred,transform_blur,patient,split,site\n"
        "0,0,0,p1,train,A\n"
        "0,0,1,p2,train,A\n"
        "1,1,0,p3,test,B\n"
        "1,1,0,p4,test,B\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.html"
    app = typer.Typer()
    register(app)

    result = CliRunner().invoke(
        app,
        [
            "dl",
            "run",
            "--data",
            str(data),
            "--task",
            "medical",
            "--patient-id",
            "patient",
            "--split",
            "split",
            "--site",
            "site",
            "--format",
            "json",
            "--out",
            str(output),
            "--no-persist",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.json").exists()
