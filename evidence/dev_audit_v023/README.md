# Development human audit evidence

These files were supplied after the completed v0.2.3 RTX 6000 Ada development
run. They cover 192 images (two scenes per category, four equivalent wordings,
two seeds, and three models) labeled by one annotator.

The audit is diagnostic and provisional. It has no second independent rater,
adjudication, or main-split evidence, so it must not be presented as the primary
ICASSP result. Version 0.2.4 uses it to justify a human-primary main protocol.

The machine-private audit key and images were not supplied with these tables.
Consequently, these files document the reported development findings but cannot
alone reconstruct or rescore the audit. To obtain the new atomic agreement fields,
run the v0.2.4 audit scorer against the original v0.2.3 run directory, which still
contains `PRIVATE_machine_key.csv`.
