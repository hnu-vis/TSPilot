# runtime/conversation_state.py SPEC

## Purpose

Define how runtime keeps short-lived conversation context across requests.

## Input

- normalized `ConversationStateModel`

## Output

- updated `ConversationStateModel`

## Responsibilities

- keep recent conversation messages
- keep latest evidence and analysis references
- preserve selected database context across turns when applicable
- retain a bounded session summary for prompt compaction

## Reads

- prior messages
- latest request outputs

## Writes

- recent messages
- latest evidence / analysis / visualization references
- database context
- session summary
- context budget hints

## Must not do

- store long-term memory
- execute business logic
