import json
from importlib.resources import as_file, files

import presentation

# -----------------------------------------------------------------------------
# Load artefacts
# -----------------------------------------------------------------------------

presentation_files = files(presentation)
artefacts_resource = presentation_files / "artefacts.json"

with as_file(artefacts_resource) as artefacts_path:
    artefacts = json.loads(artefacts_path.read_text(encoding="utf-8"))

# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------

md_content = f"""
# Power Demand Forecasting – Proof of Concept

## Executive Summary

{artefacts["executive_summary"]}

## 1. Exploratory Analysis

![Power consumption analysis]({artefacts["power_analysis_plot"]})

{artefacts["data_analysis_comment"]}

## 2. Modeling Choices

{artefacts["modeling_choices_comment"]}


## 3. Backtesting Results

### Quantitative Metrics

{artefacts["metrics_table"]}

### Forecast vs Observations

![Forecasting results]({artefacts["modeling_result_plot"]})

{artefacts["modeling_results_comment"]}

"""

# -----------------------------------------------------------------------------
# Write markdown file
# -----------------------------------------------------------------------------

output_md_resource = presentation_files / "power_forecasting_report.md"

with as_file(output_md_resource) as output_md_path:
    output_md_path.write_text(md_content.strip(), encoding="utf-8")
