# Native ECharts Visualization V5 Contract

Visualization planning is a single grounded stage:

```text
user question + Grounded Source Inventory
→ closed EChartsPlan containing option_json
→ validation and placeholder resolution
→ V5 artifact persistence
→ direct ECharts setOption(option)
```

The LLM writes native ECharts JSON but cannot write data arrays. Complete records enter only through
`{"$dataset":"source_ref"}` and scalar values through
`{"$value":{"source_ref":"source_ref","field":"field_name"}}`. The compiler derives all source refs and
evidence bindings from those placeholders.

The public artifact uses `schema_version: "5"` and `chart_type: "echarts"`. It contains the native `option`,
bindings, verification metadata, and accessibility content. The stored artifact contains complete dataset
sources; the conversational descriptor preserves the option structure but clears those sources and table rows.
The data endpoint restores the complete artifact.

Only `line`, `scatter`, and `bar` series plus native `markPoint`, `markLine`, and `markArea` are accepted.
Transforms, inline data, custom renderers, executable content, DOM/HTML, and URLs are rejected. Dataset,
encode, and axis references are validated with JSON Pointer diagnostics. Planning may be repaired by the LLM
twice; there is no deterministic fallback chart.

V4 artifacts are intentionally unsupported and are filtered when historical conversations are loaded.
