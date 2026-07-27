# Spike Pre-registration Template

Copy this file to `.docs/spikes/<spike-id>.md`. Complete every required field and commit the file before the first data-generating run. Do not revise a pre-registration after data exists; record a new, explicitly post-hoc document instead.

## Study identity

- **Spike ID and title:**
- **Evidence class:** `replication` | `adaptation` | `original`
- **Decision owner and date:**
- **First-run commit:** Leave blank until the pre-registration commit has merged; then record the first commit allowed to generate data.

## Hypothesis

State one falsifiable directional or equivalence hypothesis. Name the treatment, control, target population, and the primary comparison family.

## Primary metric

Name exactly one primary metric. Define its numerator, denominator, aggregation level, direction of improvement, and measurement procedure. Label every other metric exploratory.

## SESOI

State the smallest effect size of interest before data collection. Derive it from the user or operator decision that changes at that effect size, not from observed variance.

## Decision margins

Derive each quantity separately from utility, cost, and risk. Do not substitute one margin for another or derive any margin from observed standard deviation.

- **Minimum worthwhile gain (MWG):** State the smallest beneficial improvement worth adopting and its utility basis.
- **Non-inferiority margin (NIM):** State the maximum acceptable loss against the control and its cost basis.
- **Equivalence margin (EQM):** State a strictly positive interval around zero that is practically negligible and its utility basis.

## Clustering unit

Name the independent unit resampled by inference, explain why observations are clustered under it, and state how repeated observations remain within that cluster.

## Kill criterion

State the result or feasibility condition that stops the spike, the resulting disposition, and the evidence required to apply it. A null result is a valid outcome when declared here.

## Run and analysis boundary

Freeze the treatment matrix, inputs, seeds, exclusion rules, and analysis procedure. Record the command and immutable input revisions before the first run.

## Post-hoc changes

Record only changes made after data exists. For each, state the timestamp, reason, affected field, and why the result is exploratory.
