# Fisher Mode Intervention Report

## Baseline

- Test answer accuracy: 100.000%
- Test paired-context accuracy: 100.000%
- Test hard NLL: 0.048382

## Single-mode causal alignment

| Boundary | Fisher rank vs delta NLL | Fisher rank vs activation RMS | Activation RMS vs delta NLL |
|---|---:|---:|---:|
| `layer.0.input` | 0.7507 | 0.6397 | 0.7588 |
| `layer.0.output` | 0.8963 | 0.7397 | 0.6998 |
| `layer.1.output` | 0.8794 | 0.8607 | 0.8680 |
| `final_norm` | 0.8409 | 0.4578 | 0.4604 |

## Full-mute group comparison

| Boundary | Modes | Top delta NLL | Bottom delta NLL | Random delta NLL | Top accuracy drop |
|---|---:|---:|---:|---:|---:|
| `layer.0.input` | 1 | 0.094867 | 0.000000 | 0.019217 | 1.752% |
| `layer.0.input` | 2 | 0.408888 | 0.000103 | 0.036927 | 12.261% |
| `layer.0.input` | 4 | 0.817924 | 0.020105 | 0.119804 | 26.752% |
| `layer.0.input` | 8 | 2.069707 | 0.199103 | 0.497333 | 49.522% |
| `layer.0.input` | 16 | 3.762165 | 1.212628 | 1.652349 | 86.465% |
| `layer.0.output` | 1 | 0.047102 | -0.000000 | 0.013836 | 1.274% |
| `layer.0.output` | 2 | 0.059946 | 0.000057 | 0.038597 | 0.318% |
| `layer.0.output` | 4 | 0.353557 | 0.000280 | 0.130634 | 10.032% |
| `layer.0.output` | 8 | 2.053140 | 0.000533 | 0.323170 | 54.299% |
| `layer.0.output` | 16 | 3.555238 | 0.007623 | 1.140246 | 80.255% |
| `layer.1.output` | 1 | 0.002518 | -0.000000 | 0.009270 | 0.000% |
| `layer.1.output` | 2 | 0.063406 | 0.000073 | 0.020988 | 0.000% |
| `layer.1.output` | 4 | 0.262629 | 0.000374 | 0.055105 | 4.459% |
| `layer.1.output` | 8 | 0.618788 | 0.000822 | 0.118825 | 5.096% |
| `layer.1.output` | 16 | 2.222634 | 0.003722 | 0.507599 | 56.688% |
| `final_norm` | 1 | 0.232261 | -0.000000 | 0.037972 | 3.025% |
| `final_norm` | 2 | 0.258993 | -0.000000 | 0.067439 | 4.618% |
| `final_norm` | 4 | 0.879624 | -0.000000 | 0.152138 | 33.758% |
| `final_norm` | 8 | 2.730424 | -0.000000 | 0.399573 | 88.376% |
| `final_norm` | 16 | 2.726548 | 0.000011 | 1.083152 | 88.376% |

## Primary partial-mute comparison

| Boundary | Strength | Top delta NLL | Bottom delta NLL | Random 95% interval | Random p |
|---|---:|---:|---:|---:|---:|
| `layer.0.input` | 5% | 0.000310 | 0.000031 | [-0.000146, 0.000466] | 0.1386 |
| `layer.0.input` | 10% | 0.001193 | 0.000070 | [-0.000092, 0.001032] | 0.0198 |
| `layer.0.input` | 25% | 0.012711 | 0.000236 | [0.000401, 0.007692] | 0.0198 |
| `layer.0.input` | 50% | 0.168716 | 0.001365 | [0.004454, 0.122551] | 0.0099 |
| `layer.0.input` | 75% | 0.846340 | 0.088048 | [0.022158, 0.430905] | 0.0099 |
| `layer.0.input` | 100% | 2.069707 | 0.199103 | [0.155115, 1.213727] | 0.0099 |
| `layer.0.output` | 5% | 0.000155 | 0.000006 | [-0.000108, 0.000343] | 0.3366 |
| `layer.0.output` | 10% | 0.000843 | 0.000015 | [-0.000076, 0.001218] | 0.1287 |
| `layer.0.output` | 25% | 0.009075 | 0.000058 | [0.000518, 0.006565] | 0.0099 |
| `layer.0.output` | 50% | 0.151562 | 0.000174 | [0.004128, 0.074857] | 0.0297 |
| `layer.0.output` | 75% | 0.847164 | 0.000335 | [0.012981, 0.383058] | 0.0099 |
| `layer.0.output` | 100% | 2.053140 | 0.000533 | [0.034431, 0.870116] | 0.0099 |
| `layer.1.output` | 5% | 0.001765 | -0.000043 | [-0.000396, 0.001157] | 0.0099 |
| `layer.1.output` | 10% | 0.003919 | -0.000077 | [-0.000698, 0.002452] | 0.0099 |
| `layer.1.output` | 25% | 0.013660 | -0.000126 | [-0.000463, 0.008546] | 0.0099 |
| `layer.1.output` | 50% | 0.052379 | -0.000034 | [-0.000316, 0.033600] | 0.0099 |
| `layer.1.output` | 75% | 0.180964 | 0.000280 | [0.009759, 0.128044] | 0.0099 |
| `layer.1.output` | 100% | 0.618788 | 0.000822 | [0.025705, 0.286192] | 0.0099 |
| `final_norm` | 5% | 0.014966 | -0.000000 | [-0.000004, 0.008752] | 0.0099 |
| `final_norm` | 10% | 0.034445 | -0.000000 | [-0.000008, 0.016398] | 0.0099 |
| `final_norm` | 25% | 0.133866 | -0.000000 | [-0.000014, 0.063180] | 0.0099 |
| `final_norm` | 50% | 0.553079 | -0.000000 | [-0.000015, 0.187600] | 0.0099 |
| `final_norm` | 75% | 1.451660 | -0.000000 | [0.000011, 0.557813] | 0.0099 |
| `final_norm` | 100% | 2.730424 | -0.000000 | [0.000026, 1.113005] | 0.0099 |

### Primary cell with perturbation-energy matching

The primary cell is `layer.0.output`, top 8 modes, 25% suppression.

- Top-mode delta NLL: 0.009075
- Energy-matching strengths calibrated on: `validation_fisher`
- Energy-matched bottom delta NLL: 0.000228
- Energy-matched random 95% interval: [0.001428, 0.013689]
- Top-vs-energy-matched-random empirical p: 0.1980
- Paired-context top-minus-bottom 95% bootstrap interval: [0.007477, 0.010381]

## Modal subspace sufficiency

| Boundary | Retained subspace | Modes retained | Fisher retained | Accuracy | Hard NLL |
|---|---|---:|---:|---:|---:|
| `layer.0.input` | keep_top | 1 | 17.997% | 14.809% | 3.587467 |
| `layer.0.input` | keep_bottom | 1 | 0.000% | 11.624% | 3.643330 |
| `layer.0.input` | keep_top | 2 | 31.048% | 24.363% | 3.274160 |
| `layer.0.input` | keep_bottom | 2 | 0.187% | 11.783% | 3.671252 |
| `layer.0.input` | keep_top | 4 | 48.406% | 34.873% | 2.773701 |
| `layer.0.input` | keep_bottom | 4 | 0.645% | 11.783% | 3.688844 |
| `layer.0.input` | keep_top | 8 | 72.309% | 60.350% | 1.909365 |
| `layer.0.input` | keep_bottom | 8 | 1.945% | 12.898% | 3.711039 |
| `layer.0.input` | keep_top | 15 | 90.486% | 76.433% | 1.281778 |
| `layer.0.input` | keep_bottom | 15 | 6.845% | 13.376% | 3.878057 |
| `layer.0.input` | keep_top | 19 | 95.113% | 79.936% | 0.971899 |
| `layer.0.input` | keep_bottom | 19 | 13.252% | 16.720% | 3.517449 |
| `layer.0.input` | keep_top | 27 | 99.092% | 99.204% | 0.079319 |
| `layer.0.input` | keep_bottom | 27 | 44.021% | 65.605% | 1.176911 |
| `layer.0.input` | keep_top | 32 | 100.000% | 100.000% | 0.048382 |
| `layer.0.input` | keep_bottom | 32 | 100.000% | 100.000% | 0.048382 |
| `layer.0.output` | keep_top | 1 | 19.077% | 13.854% | 3.801930 |
| `layer.0.output` | keep_bottom | 1 | 0.000% | 11.624% | 4.750533 |
| `layer.0.output` | keep_top | 2 | 34.140% | 19.745% | 3.542673 |
| `layer.0.output` | keep_bottom | 2 | 0.076% | 11.624% | 4.725033 |
| `layer.0.output` | keep_top | 4 | 51.993% | 30.573% | 2.941448 |
| `layer.0.output` | keep_bottom | 4 | 0.308% | 11.624% | 4.643586 |
| `layer.0.output` | keep_top | 8 | 73.173% | 87.102% | 0.441536 |
| `layer.0.output` | keep_bottom | 8 | 1.224% | 9.554% | 4.369722 |
| `layer.0.output` | keep_top | 14 | 90.254% | 99.841% | 0.064689 |
| `layer.0.output` | keep_bottom | 14 | 4.407% | 15.287% | 3.634952 |
| `layer.0.output` | keep_top | 18 | 95.593% | 100.000% | 0.050775 |
| `layer.0.output` | keep_bottom | 18 | 9.746% | 22.293% | 3.205827 |
| `layer.0.output` | keep_top | 25 | 99.054% | 100.000% | 0.048927 |
| `layer.0.output` | keep_bottom | 25 | 31.059% | 54.299% | 1.827168 |
| `layer.0.output` | keep_top | 32 | 100.000% | 100.000% | 0.048382 |
| `layer.0.output` | keep_bottom | 32 | 100.000% | 100.000% | 0.048382 |
| `layer.1.output` | keep_top | 1 | 34.676% | 12.420% | 3.024613 |
| `layer.1.output` | keep_bottom | 1 | 0.000% | 11.783% | 2.964316 |
| `layer.1.output` | keep_top | 2 | 53.463% | 30.892% | 2.583636 |
| `layer.1.output` | keep_bottom | 2 | 0.014% | 11.783% | 2.953424 |
| `layer.1.output` | keep_top | 4 | 72.522% | 57.166% | 1.463233 |
| `layer.1.output` | keep_bottom | 4 | 0.065% | 11.783% | 2.906773 |
| `layer.1.output` | keep_top | 8 | 89.866% | 97.452% | 0.256421 |
| `layer.1.output` | keep_bottom | 8 | 0.264% | 18.949% | 2.802972 |
| `layer.1.output` | keep_top | 9 | 91.796% | 100.000% | 0.102215 |
| `layer.1.output` | keep_bottom | 9 | 0.344% | 18.949% | 2.758572 |
| `layer.1.output` | keep_top | 12 | 95.592% | 100.000% | 0.061001 |
| `layer.1.output` | keep_bottom | 12 | 0.691% | 19.586% | 2.657456 |
| `layer.1.output` | keep_top | 19 | 99.140% | 100.000% | 0.049642 |
| `layer.1.output` | keep_bottom | 19 | 3.580% | 58.439% | 1.880231 |
| `layer.1.output` | keep_top | 32 | 100.000% | 100.000% | 0.048382 |
| `layer.1.output` | keep_bottom | 32 | 100.000% | 100.000% | 0.048382 |
| `final_norm` | keep_top | 1 | 17.574% | 23.567% | 2.403200 |
| `final_norm` | keep_bottom | 1 | 0.000% | 11.624% | 2.774503 |
| `final_norm` | keep_top | 2 | 33.942% | 43.790% | 2.246985 |
| `final_norm` | keep_bottom | 2 | 0.000% | 11.624% | 2.774503 |
| `final_norm` | keep_top | 4 | 61.430% | 57.166% | 1.534747 |
| `final_norm` | keep_bottom | 4 | 0.000% | 11.624% | 2.774503 |
| `final_norm` | keep_top | 7 | 92.062% | 100.000% | 0.233372 |
| `final_norm` | keep_bottom | 7 | 0.000% | 11.624% | 2.774503 |
| `final_norm` | keep_top | 8 | 99.995% | 100.000% | 0.048482 |
| `final_norm` | keep_bottom | 8 | 0.000% | 11.624% | 2.774503 |
| `final_norm` | keep_top | 32 | 100.000% | 100.000% | 0.048382 |
| `final_norm` | keep_bottom | 32 | 100.000% | 100.000% | 0.048382 |

A mute fraction of 0 leaves the signal unchanged; 1 removes the
selected centered modal components completely. Modes are defined
on the validation/Fisher split, while every behavioral result
above is measured on the held-out test split.
