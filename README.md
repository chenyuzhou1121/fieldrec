# FieldRec

FieldRec re-ranks frozen sequential recommenders using evidence from the preference transitions a candidate would induce. It fits user-balanced Gaussian transition fields conditioned on history mobility, scores hypothetical rating outcomes, and applies a phase-dependent score response. It also evaluates target rankings across users and domains.

Paper: FieldRec: Prospective Preference Transitions for Frozen Recommenders (WSDM 2027).

## Installation

Use Python 3.10 or later. From the repository root, install the declared dependencies:

```sh
python -m pip install -r requirements.txt
```

NumPy handles arrays, pandas and PyArrow read Parquet tables, and SciPy supplies log-sum-exp calculations. Run the commands below from the repository root so the local modules are importable.

## Quick start

Prepare `data/transitions.parquet` and `data/candidates.npz` with the formats below. Each field and candidate bundle covers one domain.

```sh
python -m scripts.fit_field --transitions data/transitions.parquet --source train --output outputs/field.json
python -m scripts.score_candidates --inputs data/candidates.npz --field outputs/field.json --split validation --output outputs/scores.npz
python -m scripts.evaluate --scores outputs/scores.npz --min-history 5
```

Fitting requires `--transitions` and `--output`; `--source` defaults to `train`. Scoring requires `--inputs`, `--field`, `--split`, and `--output`. Both accept `--config`, defaulting to `configs/default.json`, and create output parent directories and a sibling `.log` file.

Use a `train` field for `train` or `validation` queries. For `test`, fit with `--source development`, which includes train and validation transitions, and score with `--split test`.

The model JSON contains `atlas` and `mobility`. The scored NPZ contains query metadata, `candidate_indices`, `base_scores`, `fieldrec_scores`, `potential`, `support_gate`, and `phase`. Unsupported queries keep their base scores; phases are `FAST`, `SLOW`, or `__OTHER__`.

Evaluation requires `--scores` with one or more scored NPZ paths. It prints a per-domain NDCG@5 CSV table followed by a macro JSON summary, using `null` for undefined summary values. Optional flags are `--min-history` (default 5), `--bootstrap-reps` (10000), and `--seed` (42). Metrics include MRR plus HR and NDCG at 5, 10, and 20; queries are averaged within users, users within domains, and domains equally. NDCG@5 differences receive paired user bootstrap intervals and two-sided bootstrap sign proportions.

## Input formats

### Transition Parquet

Supply chronological source transitions between observed states, including short histories, with these columns:

| Column | Meaning |
| --- | --- |
| `user_id` | User identifier within the domain. |
| `current_event_index` | Numeric chronological index of the starting event within the user. |
| `target_split` | Destination event split: `train`, `validation`, or `test`. |
| `P`, `A` | Starting preference and alignment coordinates, stored as float32. |
| `delta_P`, `delta_A` | Next coordinates minus starting coordinates, stored as float32. |

Replay observations using `fieldrec.PreferenceSufficientState.empty(D)` and `fieldrec.observe(state, item_embedding, rating, rating_baseline, residual_scale, config)`. Pass each returned `sufficient_state` into the next observation; `fieldrec.state_coordinates(state)` returns `[P, A]`. Use strictly earlier rating baselines, a positive residual scale fitted on train, fixed embeddings, and the same state settings during replay and scoring.

For clipped residual `r`, `S` accumulates `r`, `E` accumulates `abs(r)`, and `G` accumulates `abs(r)` times pre-update alignment. Alignment is half the difference between the item's cosine similarities to positive and negative memories. Coordinates are clipped `P=S/E` and `A=G/E`, or zero when mass is negligible. Positive and negative memories sum embeddings weighted by the corresponding residual magnitudes; `observe` rounds memories through float16 after each update.

### Candidate NPZ

Use numeric or Unicode arrays without pickled objects. Let Q be query count, N catalog size, D embedding dimension, and W the fitted mobility window.

| Array | Shape and meaning |
| --- | --- |
| `item_embeddings` | `(N,D)` fixed item vectors matching state replay. |
| `rating_baselines` | `(N,)` item rating baselines frozen at the field source cutoff. |
| `residual_scale` | Positive scalar rating-residual scale. |
| `candidate_indices` | `(Q,100)` unique integer catalog indices per query, zero-based. |
| `base_scores` | `(Q,100)` finite scores in candidate order; larger ranks first. |
| `sufficient` | `(Q,3)` statistics in `S,E,G` order before target feedback. |
| `positive`, `negative` | `(Q,D)` observed memories from `fieldrec.observe`. |
| `history_deltas` | `(Q,W,2)` last W finite `[delta_P,delta_A]` transitions between observed states, oldest first, NaN padded when shorter. |
| `domain`, `user_id`, `query_id` | Each `(Q,)`; identifiers for aggregation and duplicate detection. |
| `history_count` | `(Q,)` number of observed events before the query. |
| `target_position` | `(Q,)` integer target slot in the candidate slate, zero-based; used by evaluation. |

Exclude target feedback and future transitions from states and histories. Candidate scoring enumerates ratings 1 through 5 for every item. Each domain/query pair must occur once across evaluated files. Freeze response settings before test evaluation.

## Configuration

The JSON has four objects. Field fitting reads `field` and `mobility`; scoring reads `state` and `response`. The fitted model retains its field and mobility settings. All three `state`/`field`/`response` sections are required by the configuration loader, even when a command uses only some sections.

| Key | Meaning |
| --- | --- |
| `state.residual_clip` | Maximum absolute normalized rating residual, in `(0,1]`. |
| `field.grid_size` | Number of cells per coordinate axis, at least 4. |
| `field.x_range` | Increasing bounds for starting P; endpoints are included. |
| `field.y_range` | Increasing bounds for starting A; endpoints are included. |
| `field.min_shared_cell_transitions` | Minimum transition count for a shared cell. |
| `field.min_shared_cell_users` | Minimum distinct-user count for a shared cell. |
| `field.min_context_cell_transitions` | Minimum transition count for a context cell; shared support is also required. |
| `field.min_context_cell_users` | Minimum distinct-user count for a context cell. |
| `field.min_context_transitions` | Minimum total transitions to retain a named context; others are pooled. |
| `field.min_context_users` | Minimum distinct users to retain a named context. |
| `field.shrinkage_prior_ess` | Positive parent strength; child weight is `ESS/(ESS+prior)`. |
| `field.covariance_regularization` | Positive diagonal addition to covariance matrices. |
| `field.covariance_shrinkage` | Weight in `[0,1]` for shrinkage toward the variance-floored diagonal. |
| `field.min_variance` | Positive floor for diagonal targets and covariance eigenvalues. |
| `field.max_fallback_cells` | Maximum Euclidean distance in cell units for nearest supported-cell fallback. |
| `response.kappa_fast` | FAST phase sign: `-1`, `0`, or `1`; zero disables its response. |
| `response.kappa_slow` | SLOW phase sign: `-1`, `0`, or `1`; zero disables its response. |
| `response.beta` | Nonnegative finite response multiplier. |
| `response.tau` | Activation threshold in `[0,1]` for the signed centered percentile. |
| `mobility.window` | Maximum number of prior finite delta norms averaged per state. |
| `mobility.min_prior` | Minimum prior count for a mobility label, between 1 and `window`. |
| `mobility.threshold_quantile` | User-balanced fitting quantile in `(0,1)` separating SLOW from FAST. |

Mobility excludes the current transition. Scores at or below its fitted threshold are SLOW; larger scores are FAST. Too little history yields `__OTHER__`. Supported candidate potentials are log ratios of local and context-global densities summed over the five rating branches. The score addition is `beta * gate * max(kappa * x - tau, 0)`, where `x` is the candidate's centered average-rank percentile and `gate` is binary support.

## Directory structure

- `fieldrec/`: state replay, covariance utilities, field models, mobility labels, scoring, response, and evaluation.
- `scripts/`: the three command-line entry points shown above.
- `configs/default.json`: default state, field, response, and mobility settings.
- `requirements.txt`: third-party Python dependencies.
- `LICENSE`: license text.

## License

MIT. See [LICENSE](LICENSE).
