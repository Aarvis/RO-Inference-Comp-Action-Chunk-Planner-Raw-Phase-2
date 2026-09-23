# Vendored OpenPI Source

This directory vendors the OpenPI runtime source required by
`pi05_origami_comp_action_chunk`.

Expected layout:

```text
vendor/openpi/src/
  openpi/
  openpi_client/
  future_latent_predictor/
```

The vendored source must match the OpenPI checkout used for training the final
policy checkpoint. The model bundle should contain only runtime assets, configs,
weights, and optional dataset replay data; it should not contain an extra
`code/openpi` copy.
