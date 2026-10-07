# Apertus v1.5 booklet-grounded research protocol

## Fixed task and inputs

Follow challenge_contract.md. Production receives a voting booklet PDF and claim
only. Classes are 0 entailment, 1 neutral, 2 contradiction. DE/FR/IT and all nine
language pairs are required. References are oracle diagnostics, never production
inputs. CLI/evidence schemas are project choices; Devpost is unnecessary.

Use the pinned native Apertus v1.5 revision a411d838600baf0e3635a3daf66fb7c55fc97bb6
and official Transformers fork 3797303dda74844e3d1f8977ff5518bb91f818b4. Verify every
original weight shard before loading. BF16 text and supported eager vision/audio
tokenizers are used without quantization. Cached weights require no HF credential.
The current model weights remain frozen; a trained decision head is distinct from
Apertus fine-tuning or LoRA.

The canonical split is 902 train / 276 validation / 310 internal holdout. Grouping
connects voting dates sharing normalized identical claims. All source PDFs must
match pinned original hashes; no damaged or replacement PDF is accepted. Validation
and test must be complete and fingerprint-identical. If public source availability
still removes training PDFs, only the explicit bootstrap training-subset flag may
be used: canonical IDs/claims/labels/groups remain unchanged, exclusions must match
the PDF audit, and the exact selected/missing IDs and count must be reported.
Selection depends on source availability, never training difficulty or test labels.

## Predeclared comparison and selection

1. Prepare plain-extraction full-booklet rows with source pages, offsets and SHA-256.
2. Freeze full/capped, oracle/reference, BM25, E5 dense and hybrid k=5 contexts.
   References are available only for the validation oracle experiment. Retrieval
   reads booklet and claim, never the label/reference. Record byte/token truncation.
3. On all 276 validation IDs, compare native restricted class scoring and prompted
   JSON decisions for each context. Record Macro-F1, class/language/pair/cross-language
   metrics, tokens and model/retrieval time. Invalid prompt outputs are reported
   separately; a subset score is never presented as full output coverage.
4. Select the highest validation Macro-F1 booklet-only class-score context, breaking
   ties by fewer tokens. Exclude oracle contexts from production selection.
5. Extract frozen Apertus option-logit and final hidden features for training and
   that validation context. Fit two standardized logistic heads on train only, with
   C in {0.001, 0.01, 0.1, 1}. Use up to five document/duplicate-claim grouped OOF
   folds, requiring all three classes in fitting folds. Select C and temperature
   from training OOF results; report actual group counts and source limitations.
6. Compare both heads with the best complete valid booklet-only validation baseline.
   The fixed head GO gate requires at least +0.02 Macro-F1. Otherwise retain the
   best baseline. No LoRA, weight update or additional model is implied by this gate.
7. Freeze the selected model/context/decision method and evaluate all 310 internal
   holdout IDs once for NLI. Never optimize against this result. This is not the
   organizer's hidden benchmark or an official challenge score.
8. Exercise the actual public PDF CLI on one holdout example per claim language,
   with PDF + claim only. Require labels to match frozen predictions and exact
   quote/page provenance. Export source-checksummed results before lease destruction.

The existing 5-gram reference-overlap retrieval diagnostics are not organizer
semantic evidence scores, exact gold Recall@k, or NLI accuracy. Latency excludes
first download, model loading and PDF parsing; separate retrieval and model timing
is retained. Small independent group counts limit generalization and calibration.

## Execution and artifacts

Public scripts implement the sequence: remote_v15_bootstrap.py,
prepare_booklets.py, prepare_v15_inputs.py, vast_v15_worker.sh,
run_v15_experiments.py, train_v15_head.py and finish_v15_research.py.
The successful source revision and script hashes are recorded in each artifact.
Outputs are immutable; existing experiment directories cannot be overwritten.
Completed research exports to experiments/apertus-v15-research-v1 with per-file
SHA-256 and a source-checksummed archive. Binary features/raw data/weights are ignored
by Git; JSON heads, predictions, metrics, manifests and negative results are retained.
Integration checks in v15-access-v1, v15-real-cpu-cli-v1, v15-real-docker-v1 and
v15-fresh-docker-v1 are real execution checks, not benchmark estimates.

## Budget and cleanup

Use the official Vast API through the existing VAST_API_KEY binding. Never print
credentials, dump environments, add keys to source, or pass them in SSH commands.
Before any rental inspect account balance, task-owned active leases, current offer,
VRAM, storage/network charges and a bounded automatic destruction deadline.
The total hard budget is USD 10 with USD 2 reserve. A stopped lease still retains
storage charges; destroy it after artifacts/caches are verified elsewhere, and
verify absence using paginated account instance listings. Never destroy unrelated
user leases. Source/cache recovery may use private provider copy without HF keys;
wait for destination readiness, then verify all original SHAs. In this task a running
container needed an official reboot before copied files became visible; stopping
and restarting is unsafe because resources may become unavailable. The controller
must still verify results and destroy the final stopped research lease.
Billing can settle asynchronously; report timestamped observed credit reduction and
active reservations separately. Current status is experiments/budget.json, not any
archived lease description. Legacy results and scope assumptions are preserved in
legacy_experiment_protocol.md and legacy_apertus_2509_report.md.
