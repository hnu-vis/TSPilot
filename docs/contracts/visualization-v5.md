# Visualization

## Overview

Visualization turns existing time-series data and analytical Insights into an
interactive chart. Its job is to help users verify a conclusion visually, not
to perform the analysis again.

The current visualization flow is:

```text
User question
    ↓
Describe the available data and Insights
    ↓
Choose the most useful line and annotations
    ↓
Validate that every visual element has supporting data
    ↓
Render an interactive ECharts line chart
```

## What the visualization prompt receives

The prompt does not receive an unexplained collection of raw objects. It gets a
compact description of the information already produced upstream.

Each available dataset is described by:

- what the data represents and what it can be used to show;
- its time range and number of records;
- the time and numeric fields available for plotting;
- a small data example;
- where it came from and which Insights it supports;
- any known limitations.

The example rows only help the model understand the shape of the data. The
rendered chart always uses the complete dataset.

Each Insight is described by:

- the conclusion itself;
- its result and unit;
- how the result was calculated;
- important operands, times, intervals, or selected points;
- the datasets that can visually support the conclusion.

This description reduces ambiguity without requiring changes to upstream tool
outputs. It also avoids repeating the same evidence in several different forms.

## How a chart is chosen

The model chooses a small semantic chart plan rather than writing ECharts JSON
directly. It decides:

- which full time series best answers the question;
- whether a second compatible line is needed for comparison;
- which Insight points, intervals, or levels are important enough to mark;
- the chart title, short summary, and axis label.

The system then turns that plan into the final ECharts configuration. This keeps
data binding, styling, legends, tooltips, and zoom behavior consistent across
questions.

The current production output is one primary line chart with at most two line
series. A second line is used only when it is necessary for a meaningful
comparison.

## Data and Insight roles

The chart uses full time-series data to show the observed trajectory. Insights
are used to explain and annotate that trajectory.

Typical examples include:

- a lowest or highest point shown on the full price line;
- a strongest rise or fall shown with its start, end, and time interval;
- an average or threshold shown as a reference line;
- two compatible monthly or period series shown together for comparison.

A calculated difference, percentage, duration, or count is normally kept in the
title or summary. It is not drawn as a price level unless upstream data has
already provided an appropriate series for it.

Sparse calculation results are not presented as if they were a complete
trajectory. For example, two rebound endpoints may be marked on the full price
line, but they are not connected and described as the observed price path.

## Forecast and anomaly results

Forecast and anomaly tools enter visualization through the same data
description layer:

- forecast points may be used as a line;
- forecast intervals and quality information provide supporting context;
- anomaly scores may be shown as a time series when appropriate;
- anomaly points and spans can support located findings and annotations.

Whether a result becomes a line or an annotation depends on its meaning and
shape. Scalar status or quality information is not forced into a time-series
chart.

## Chart behavior

Every rendered chart includes:

- a visible legend for ordinary line series;
- tooltips for inspecting values;
- mouse/touch zoom and a visible range slider;
- a separate legend for Insight points, intervals, and reference lines when
  those annotations exist;
- an accessible description and a compact supporting data table.

The chart title appears once in the surrounding card. ECharts does not render a
second duplicate title inside the plot.

## Grounding and failure handling

Every line and annotation must resolve to existing data or a verified Insight.
The system rejects charts that use unknown fields, invent annotation values,
mix incompatible scales, omit required comparison data, or turn sparse
endpoints into a misleading trend.

If the requested chart needs data that has not yet been produced, visualization
asks the appropriate upstream tool—such as query, calculation, forecast, or
anomaly detection—to provide it. The ReAct loop can then retry visualization
with the new result.

If a proposed chart is invalid, the model receives a concise explanation and
regenerates the chart plan. There is one initial attempt and up to two repair
attempts. The system does not create a substitute chart with hard-coded logic;
if the chart still cannot be grounded, visualization reports that it is
unavailable.

## Storage and display

The complete visualization is stored as a V5 ECharts artifact. Conversation
state keeps a lightweight version, and the frontend loads the full data only
when it needs to render the chart.

Older unsupported visualization formats are not silently converted. The
frontend reports them as unavailable instead of guessing how they should be
displayed.

## Testing visualization independently

Visualization can be tested without rerunning database queries and upstream
analysis. The replay script loads a previous request state, keeps its data and
Insights, reruns only visualization, and captures the resulting charts:

```bash
/home/feilvvl/TSPilot/tspilot_env/bin/python \
  scripts/replay_visualization_from_logs.py \
  --log-root cache_data/conversation_logs \
  --output-root artifacts/visualization_replay \
  --frontend http://127.0.0.1:5173
```

Generated reports and screenshots are written under the ignored `artifacts/`
directory.
