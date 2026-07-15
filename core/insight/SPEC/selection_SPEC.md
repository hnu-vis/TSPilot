# core/insight/selection.py SPEC

## Purpose

Select the most relevant verified facts for the current request.

## Responsibilities

- rank verified facts by relevance and evidence strength
- reduce duplicates
- keep the selected set aligned with the user's request

## Must not do

- change verification results
- invent new facts
