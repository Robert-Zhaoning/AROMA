# AROMA 2.0 — CSA Actuation-Strength Audit v1

Status: COMPLETED DIAGNOSTIC

Population:

653 frozen non-NOOP Proc-Count-Causal v3 examples.

Definition:

rho(x) =
||U4 U4^T h(x)|| / ||h(x)||

where:

h(x) = mean_t H_t^(18,13).

Observed distribution:

q00 = 0.05428614195457686
q10 = 0.23263106372492684
q25 = 0.2805264884148626
q50 = 0.30857232363070414
q75 = 0.33928740876382396
q90 = 0.3790007170556846
q100 = 0.510192490162274

mean rho:

0.30815994034694233

median projected activation energy share:

0.09521687891085201

Thus the median rank-4 projection contains approximately
9.52% of activation energy, even though Gate A found that
the rank-4 subspace contains approximately 89.8% of total
cardinality sensitivity and 86.8% of centered cardinality
sensitivity.

Interpretation:

The same-alpha CSA experiment is substantially weaker in
pooled intervention magnitude than whole-head scaling.

For a given alpha:

||Delta h_CSA||
/
||Delta h_whole||

=
rho(x).

For the dominant alpha=4 group, median rho is approximately
0.305.

This motivates a pre-specified magnitude-matched CSA
experiment.

No behavioral outcome from magnitude-normalized CSA has
been observed at the time of this record.
