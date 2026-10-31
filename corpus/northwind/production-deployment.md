---
title: Production Deployment Policy
owner: Platform Engineering
groups: [engineering]
---
# Production Deployment Policy

## Change windows
Production deployments are allowed Monday to Thursday between 08:00 and 16:00. Deployments on Fridays, weekends and public holidays require approval from the Head of Platform Engineering, except for SEV1 fixes.

## Requirements
Every change requires a merged pull request with at least one approving review, green CI and a rollback plan. Database migrations must be backward compatible with the previous application version.

## Progressive delivery
Services are rolled out with a canary: 5% of traffic for 30 minutes, then 50% for 30 minutes, then 100%. The rollout is stopped automatically if the error rate exceeds 1% or p95 latency increases by more than 20% compared to the baseline.

## Freeze periods
A deployment freeze applies from 20 December to 6 January.
