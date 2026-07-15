# app/routes/chat.py SPEC

## Purpose

Expose the main chat endpoint.

## Input contract

- request body matching `schemas.api.ChatRequest`

## Output contract

- response body matching terminal `schemas.api.ChatResponse`
- optional stream events

## Responsibilities

- validate the HTTP request
- normalize legacy database aliases into `database_context`
- build the initial request context
- call `data_agent`
- return the terminal payload: final answer or error

## Must not do

- embed ReAct logic
- embed tool orchestration
- infer facts directly
