# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the AlfWorld implementation from the Reflexion paper (NeurIPS 2023): "Reflexion: Language Agents with Verbal Reinforcement Learning". The codebase implements agents that learn to complete household tasks in the AlfWorld text-based environment through iterative self-reflection.

Paper: https://arxiv.org/abs/2303.11366

## Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (required)
export OPENAI_API_KEY=<your key>

# Set AlfWorld data path (required by base_config.yaml)
export ALFWORLD_DATA=<path to alfworld data>
```

## Running Experiments

### Reflexion Run (with memory)
```bash
./run_reflexion.sh
```
Runs agents with self-reflection memory across multiple trials. Default: 10 trials, 134 environments, using GPT-3.5-turbo.

### Baseline Run (without memory)
```bash
./run_simple.sh
```
Runs agents without memory/reflection for baseline comparison.

### Custom Run
```bash
python main.py \
  --num_trials 10 \
  --num_envs 134 \
  --run_name "my_run_name" \
  --use_memory \
  --model "gpt-3.5-turbo"  # or "gpt-4" or "text-davinci-003"
```

### Resume Interrupted Run
```bash
python main.py \
  --num_trials 10 \
  --num_envs 134 \
  --is_resume \
  --resume_dir "path/to/logs" \
  --start_trial_num 5
```

## Architecture

### Core Loop (main.py:28-114)

1. **Trial Loop**: Iterates through trials (learning episodes)
2. **Environment Execution**: For each trial, runs all environments via `run_trial()`
3. **Memory Update**: After each trial, generates reflections via `update_memory()` (if `--use_memory`)
4. **Logging**: Saves environment configs and trial results

### Key Components

**alfworld_trial.py**: Core trial execution
- `run_trial()`: Executes one trial across all environments
- `alfworld_run()`: Runs a single environment episode with LLM agent
- `llm()`: Wrapper for OpenAI API calls with temperature-based retries
- Task-specific prompts loaded from `prompts/alfworld_3prompts.json`
- PREFIXES dict maps task types (pick_and_place, pick_clean_then_place, etc.) to prompt keys

**generate_reflections.py**: Self-reflection generation
- `update_memory()`: Processes trial logs and generates reflections for failed tasks
- `_generate_reflection_query()`: Creates prompt for LLM to reflect on failures
- Uses few-shot examples from `reflexion_few_shot_examples.txt`
- Only keeps last 3 reflections per environment to prevent context overflow

**env_history.py**: Environment interaction tracking
- `EnvironmentHistory`: Maintains query, observation/action history, and exhaustion state
- Detects when agent repeats same action (exhaustion check in env_history.py:18)
- Formats history with reflections from previous attempts

**utils.py**: OpenAI API utilities
- `get_completion()`: For text-davinci-003 model
- `get_chat()`: For GPT-4 and GPT-3.5-turbo chat models
- Uses tenacity retry with exponential backoff

### Environment Configuration Structure

Each environment maintains state across trials:
```python
{
    'name': 'env_0',
    'memory': [reflection1, reflection2, ...],  # Self-reflections from past failures
    'is_success': False,  # Whether task was completed
    'skip': False  # Whether to skip this env
}
```

### Prompt Structure

Base prompts contain 2 few-shot examples per task type (react_{task}_0 and react_{task}_1).

When using memory, the agent receives:
- Base prompt with examples
- "Your memory for the task below:"
- Last 3 reflections from previous failed attempts
- Current task description

### Task Types (PREFIXES mapping)

- `pick_and_place` → 'put'
- `pick_clean_then_place` → 'clean'
- `pick_heat_then_place` → 'heat'
- `pick_cool_then_place` → 'cool'
- `look_at_obj` → 'examine'
- `pick_two_obj` → 'puttwo'

### Logging Structure

```
<run_name>/
├── world.log                      # High-level trial results
├── trial_0.log                    # Detailed action/observation log for trial 0
├── env_results_trial_0.json       # Environment configs after trial 0
├── trial_1.log
├── env_results_trial_1.json
└── ...
```

Pre-existing logs available in:
- `root/base_run_logs/` - Baseline runs without memory
- `root/reflexion_run_logs/` - Reflexion runs with memory

### Key Implementation Details

1. **Action Loop**: Max 49 steps per environment (alfworld_trial.py:56)
2. **Think Actions**: "think:" actions receive "OK." observation (alfworld_trial.py:62)
3. **Memory Limit**: Last 3 reflections used to prevent context overflow (alfworld_trial.py:47, generate_reflections.py:39)
4. **Success Skipping**: Successful environments skip execution in future trials (alfworld_trial.py:112-120)
5. **Exhaustion Detection**: Repeating same action triggers early termination (env_history.py:18-21)
6. **Temperature Retry**: LLM calls retry up to 6 times with increasing temperature (0.0, 0.2, 0.4...) if output too short (alfworld_trial.py:26-33)

### AlfWorld Configuration (base_config.yaml)

- Uses TextWorld environment (`AlfredTWEnv`)
- Eval split: `eval_out_of_distribution`
- 6 task types enabled
- Oracle controller for object detection
- Expert timeout: 150 steps
