# LineChart Visualization V4

`VisualizationPayload` is one grounded LineChart. It is renderer-independent
and contains typed `data_views`, axes, standard LineChart components, evidence
bindings, interactions, and accessibility data.

Supported components are `lines`, `points`, `bands`, `intervals`,
`reference_lines`, and `annotations`. Every component references a real
`view_id` and source field. Annotations use closed chart/x/xy/interval targets;
reference lines must share the target y-axis measure and unit.

The complete artifact is persisted by `VisualizationArtifactStore`. Final
answers carry a descriptor with records removed and a `data_ref` for hydration.
Old V3 and pre-LineChart V4 artifacts are not migrated or interpreted.
