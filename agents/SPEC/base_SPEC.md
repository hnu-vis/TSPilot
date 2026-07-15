# agents/base.py SPEC

## Purpose

Define the stable agent interface.

## Responsibilities

- expose a single request/response contract for the outer agent
- carry prompt building and action parsing hooks

## Must not do

- assume multiple agents
- implement tool logic
