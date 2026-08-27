# Renormalization

The workflow inspects target, denominator, and factor coordinates before applying any scheme.
Ratio and MSbar external-denominator operations are pointwise sample-wise
division. Hybrid operations switch at the authored physical distance and must
preserve sample correlations and continuity. Self-renormalization fitting and
application are selected by the explicit `type`. Each self-renormalization job
uses its authored `kernel_id`; do not infer a formula from the target observable
or from the fitted factor. The application tool is terminal.
