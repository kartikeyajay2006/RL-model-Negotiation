# DealForge OpenEnv Environment

Production-style negotiation environment implemented with OpenEnv `MCPEnvironment`.

## What is implemented

- Core environment class: `DealForgeMCPEnvironment`
- OpenEnv lifecycle methods: `reset()`, `step(action)`, `state` (property), `close()`
- Dynamic state dictionary with:
  - `negotiation_history`
  - `current_offer`
  - `time_steps`
  - `risk_score`
  - `trust_score`
  - `external_pressure`
  - `visible_state`
  - `hidden_state`
- Modular internal agents:
  - `SellerAgent`
  - `LegalAgent`
  - `RiskAgent`
  - `CompetitorAgent`
- Config-driven stochastic dynamics (no fixed outcomes)
- Dedicated multi-component reward engine with delayed effects

## Files

- `dealforge/openenv_environment.py` - main MCP environment
- `dealforge/openenv_models.py` - Action/Observation/State models
- `dealforge/internal_agents.py` - modular internal agents
- `dealforge/reward_engine.py` - weighted reward decomposition + delayed penalties
- `configs/dealforge_default.yaml` - parameterized configuration
- `examples/run_openenv_dealforge.py` - interaction flow example

## Install

```bash
python -m pip install openenv-core pyyaml
```

## Run environment example

```bash
python examples/run_openenv_dealforge.py
```

## TRL/GRPO training pipeline (Colab-ready)

Install:

```bash
python -m pip install "trl>=1.0.0" "transformers>=4.46.0" "datasets>=2.20.0" peft accelerate bitsandbytes matplotlib pyyaml
```

Train + baseline/trained comparison + plots + transcripts:

```bash
python training/train_dealforge_grpo_colab.py \
  --config configs/dealforge_default.yaml \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/dealforge_grpo_run \
  --train_prompts 128 \
  --eval_episodes 40 \
  --max_steps 120
```

Standalone evaluation:

```bash
python training/evaluate_dealforge_policy.py \
  --model outputs/dealforge_grpo_run/trained_model \
  --config configs/dealforge_default.yaml \
  --output outputs/dealforge_grpo_run/eval_trained.json
```

Regenerate plots from metrics JSON:

```bash
python training/plot_training_results.py \
  --baseline outputs/dealforge_grpo_run/baseline_metrics.json \
  --trained outputs/dealforge_grpo_run/trained_metrics.json \
  --out_dir outputs/dealforge_grpo_run
```

Artifacts:

- `baseline_metrics.json`
- `trained_metrics.json`
- `comparison.json`
- `reward_over_time.png`
- `success_quality_comparison.png`
- `transcripts_baseline.jsonl`
- `transcripts_trained.jsonl`
- `summary.md`

