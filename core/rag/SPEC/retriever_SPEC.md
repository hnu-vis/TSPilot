# core/rag/retriever.py SPEC

## Purpose

Retrieve and rank external knowledge passages.

## Responsibilities

- retrieve relevant passages
- normalize retrieval output
- expose summarized context for `rag`

## Must not do

- write final answer text
- query databases directly
