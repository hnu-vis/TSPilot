# app/server.py SPEC

## Purpose

Build and configure the FastAPI application.

## Responsibilities

- register routes
- register middleware
- load dependency wiring
- attach startup / shutdown hooks

## Must not do

- execute agent logic
- perform tool orchestration
- contain business analysis code
