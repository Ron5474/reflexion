# Reflexion Research Notes

This document provides a deep dive into the Reflexion system architecture, the three core components, and how performance is measured in the AlfWorld implementation.

## Table of Contents
- [The Three Components](#the-three-components)
- [How They Work Together](#how-they-work-together)
- [Performance Measurement](#performance-measurement)
- [Analysis and Evaluation](#analysis-and-evaluation)
- [Potential Improvements](#potential-improvements)

---

## The Three Components

The Reflexion architecture consists of three key components that work in a feedback loop:

### 1. ACTOR: The Action-Generating Agent

**Location:** `alfworld_trial.py:46-72` in `alfworld_run()`

The Actor is the LLM that generates actions to solve tasks in the environment.

```python
def alfworld_run(env, base_prompt, memory: List[str], to_print=True, ob='', model: Model):
    env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])

    cur_step = 0
    while cur_step < 49:
        # THIS IS THE ACTOR - LLM generates next action
        action = llm(str(env_history) + ">", stop=['\n'], model=model).strip()

        env_history.add("action", action)
        observation, reward, done = env.step([action])

        if done:
            return env_history, True
```

**Implementation:**
- `llm()` function call at alfworld_trial.py:57
- Takes current state (prompt + history + memory of past reflections)
- Generates next action (e.g., "go to cabinet 1", "take apple 2 from countertop")
- Uses same models as self-reflection: GPT-4, GPT-3.5-turbo, or text-davinci-003

**Input Format:**
```
[Base prompt with 2 few-shot examples]

Your memory for the task below:
Trial 0: I was stuck in a loop examining stoveburner 1...
Trial 1: I should check containers before countertops...

Here is the task:
You are in the middle of a room...
Your task is to: put some apple in fridge.
> go to fridge 1
The fridge 1 is closed.
> open fridge 1
...
> [NEXT ACTION GENERATED HERE]
```

### 2. EVALUATOR: Success/Failure Detector

**Location:** Two-level evaluation system

#### Environment-Level Evaluator (alfworld_trial.py:67-70)
```python
observation, reward, done, info = env.step([action])
observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]

if done:
    return env_history, True  # Task succeeded!
elif env_history.check_is_exhausted():
    return env_history, False  # Task failed (stuck in loop)
```

#### Trial-Level Evaluator (alfworld_trial.py:125-134)
```python
final_env_history, is_success = alfworld_run(env, base_prompt, ...)

if is_success:
    status_str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
    env_configs[z]['is_success'] = True
    num_successes += 1
    num_additional_successes += 1
else:
    status_str = f'Environment #{z} Trial #{trial_idx}: FAIL'
```

**Evaluation Criteria:**
1. ✅ **Success**: `done=True` from AlfWorld environment (task completed correctly)
2. ❌ **Failure Cases:**
   - Max 49 steps reached (alfworld_trial.py:56)
   - Agent repeats same action (stuck in loop - env_history.py:18-21)
   - Environment returns `done=True, won=False`

**Exhaustion Detection** (env_history.py:18-21):
```python
def add(self, label: str, value: str):
    if label == 'action':
        if value == self._last_action:
            self._is_exhausted = True  # Repeated action = stuck
        else:
            self._last_action = value
```

### 3. SELF-REFLECTION: Reflection Generator

**Location:** `generate_reflections.py:29-47` in `update_memory()`

The Self-Reflection component analyzes failures and generates verbal reflections to guide future attempts.

```python
def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]):
    """Called after each trial to generate reflections"""
    with open(trial_log_path, 'r') as f:
        full_log = f.read()

    env_logs = full_log.split('#####\n\n#####')

    for i, env in enumerate(env_configs):
        # ONLY reflect on failures
        if not env['is_success'] and not env['skip']:
            memory = env['memory'][-3:] if len(env['memory']) > 3 else env['memory']

            # Generate reflection query
            reflection_query = _generate_reflection_query(env_logs[i], memory)

            # THIS IS THE SELF-REFLECTION MODEL
            reflection = get_completion(reflection_query)

            # Store reflection in memory
            env_configs[i]['memory'] += [reflection]

    return env_configs
```

**Reflection Prompt Structure** (generate_reflections.py:12-27):
```python
def _generate_reflection_query(log_str: str, memory: List[str]) -> str:
    scenario = _get_scenario(log_str)

    query = f"""You will be given the history of a past experience in which you
    were placed in an environment and given a task to complete. You were unsuccessful
    in completing the task. Do not summarize your environment, but rather think about
    the strategy and path you took to attempt to complete the task. Devise a concise,
    new plan of action that accounts for your mistake with reference to specific
    actions that you should have taken.

    {FEW_SHOT_EXAMPLES}
    {scenario}"""

    if len(memory) > 0:
        query += '\n\nPlans from past attempts:\n'
        for i, m in enumerate(memory):
            query += f'Trial #{i}: {m}\n'

    query += '\n\nNew plan:'
    return query
```

**Key Properties:**
- Uses same LLM as Actor (just different prompts)
- Only generates reflections for FAILED tasks
- Uses few-shot examples from `reflexion_few_shot_examples.txt`
- Keeps only last 3 reflections to prevent context overflow
- Reflections stored in `env_configs[i]['memory']` list

**Example Reflection Output:**
```
"I was stuck in a loop in which I continually examined stoveburner 1 instead of
heating mug 1 with stoveburner 1. I should have taken mug 1 from countertop 1,
then heated it with stoveburner 1, then put it in coffeemachine 1. It did not
help to execute two identical actions in a row. I will try to execute a different
action if I am stuck in a loop again."
```

---

## How They Work Together: The Reflexion Loop

### Master Loop (main.py:86-114)

```python
trial_idx = args.start_trial_num
while trial_idx < args.num_trials:
    # STEP 1: ACTOR + EVALUATOR in action
    run_trial(trial_log_path, world_log_path, trial_idx, env_configs,
              use_memory=args.use_memory, model=args.model)

    # STEP 2: SELF-REFLECTION generates reflections for failures
    if args.use_memory:
        env_configs = update_memory(trial_log_path, env_configs)

    # STEP 3: Save updated configs (with new reflections)
    with open(trial_env_configs_log_path, 'w') as wf:
        json.dump(env_configs, wf, indent=4)

    trial_idx += 1
```

### Detailed Flow for ONE Environment Across Trials

```
TRIAL 0:
├─ ACTOR: LLM generates actions
│  ├─ Input: Base prompt + task + memory=[]
│  ├─ Output: "go to countertop 1" → "take mug 1" → "go to stoveburner 1"
│  │          → "examine stoveburner 1" → "examine stoveburner 1" → ...
│  └─ Gets stuck examining same location repeatedly
│
├─ EVALUATOR: Detects failure
│  ├─ check_is_exhausted() = True (repeated "examine stoveburner 1")
│  ├─ Returns (env_history, is_success=False)
│  └─ env_configs[0]['is_success'] = False
│
└─ SELF-REFLECTION: Generates reflection
   ├─ Input: Failed trajectory + empty memory
   ├─ LLM analyzes: "I examined stoveburner repeatedly instead of heating"
   └─ Output: "I was stuck in a loop... I will execute different actions if stuck"
   └─ env_configs[0]['memory'] = [reflection_0]

TRIAL 1:
├─ ACTOR: LLM generates actions WITH memory
│  ├─ Input: Base prompt + task + memory=["I was stuck in a loop..."]
│  ├─ Agent sees its past mistake in context
│  ├─ Output: "go to countertop 1" → "take mug 1" → "go to stoveburner 1"
│  │          → "heat mug 1 with stoveburner 1" → "go to coffeemachine 1"
│  │          → "put mug 1 in/on coffeemachine 1"
│  └─ Success!
│
├─ EVALUATOR: Detects success
│  ├─ done=True from environment
│  ├─ Returns (env_history, is_success=True)
│  └─ env_configs[0]['is_success'] = True
│
└─ SELF-REFLECTION: Skipped
   └─ No reflection generated for successful tasks
   └─ env_configs[0]['memory'] unchanged

TRIAL 2:
└─ Environment skipped (already successful)
   └─ env_config['is_success']=True, so continue without running
```

### Environment Configuration State

Each environment maintains persistent state across trials:
```python
{
    'name': 'env_0',
    'memory': [
        "Trial #0: I was stuck in a loop examining stoveburner 1...",
        "Trial #1: I should check fridge before cabinets for food items...",
        "Trial #2: I forgot to clean the apple before putting in sidetable..."
    ],
    'is_success': False,  # Whether task was ever completed
    'skip': False         # Whether to skip this env
}
```

### Same LLM, Different Prompts

**Important:** Both Actor and Self-Reflection use the same underlying LLM:

```python
# utils.py - Shared API calls
get_completion(prompt, ...)  # For text-davinci-003
get_chat(prompt, model, ...)  # For gpt-4, gpt-3.5-turbo

# Actor call (alfworld_trial.py:57)
action = llm(str(env_history) + ">", stop=['\n'], model=model)

# Self-Reflection call (generate_reflections.py:44)
reflection = get_completion(reflection_query)
```

The difference is in the **prompt engineering**, not the model.

---

## Performance Measurement

### Core Metrics Tracked

#### 1. Per-Trial Metrics (alfworld_trial.py:147-159)

After each trial, the system logs:

```python
log_str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
```

**Metric Definitions:**
- `num_successes`: **Cumulative** count (includes previously solved + newly solved)
- `num_additional_successes`: **NEW** successes in this trial only (key learning metric!)
- `num_fails`: Number of still-unsolved environments
- `accuracy`: Success rate = num_successes / total_envs

**Why Additional Success Matters:**
- Shows whether agent is actually *learning* from reflections
- If additional_success=0 for several trials → agent not improving
- Tracking this across trials gives the learning curve

#### 2. Success Determination Logic (alfworld_trial.py:101-161)

```python
num_successes = 0
num_additional_successes = 0
num_envs = len(env_configs)

for z, env_config in enumerate(env_configs):
    # Skip environments already solved
    if env_config["is_success"]:
        num_successes += 1
        with open(world_log_path, 'a') as wf:
            wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
        continue

    # Run unsolved environment
    final_env_history, is_success = alfworld_run(env, base_prompt,
                                                  env_config["memory"], ...)

    # Update metrics based on result
    if is_success:
        env_configs[z]['is_success'] = True
        num_successes += 1
        num_additional_successes += 1  # This is a NEW success!
    else:
        status_str = f'Environment #{z} Trial #{trial_idx}: FAIL'
```

**Key Insight:** Once an environment succeeds (`is_success=True`), it's skipped in all future trials. This means:
- `num_successes` carries forward and can only increase
- The agent doesn't waste time re-solving already-solved tasks
- Focus shifts to remaining difficult environments

### Example Performance Trajectory

```
***** Start Trial #0 *****
Environment #0 Trial #0: FAIL
Environment #1 Trial #0: SUCCESS
Environment #2 Trial #0: FAIL
Environment #3 Trial #0: FAIL
... (130 more environments)
-----
SUCCESS: 45          # 45/134 solved on first try
ADDITIONAL SUCCESS: 45
FAIL: 89
TOTAL: 134
ACCURACY: 0.34       # 34% success rate
-----

***** Start Trial #1 *****
Environment #0 Trial #1: SUCCESS  # Learned from reflection!
Environment #1 Trial #1: SUCCESS  # Skipped (already done)
Environment #2 Trial #1: FAIL     # Still struggling
Environment #3 Trial #1: SUCCESS  # Learned from reflection!
...
-----
SUCCESS: 67          # 67/134 now solved (45 + 22 new)
ADDITIONAL SUCCESS: 22  # 22 new successes this trial
FAIL: 67
TOTAL: 134
ACCURACY: 0.50       # 50% success rate
-----

***** Start Trial #2 *****
...
-----
SUCCESS: 85
ADDITIONAL SUCCESS: 18  # Learning is slowing down
ACCURACY: 0.63
-----

...

***** Start Trial #9 *****
...
-----
SUCCESS: 108
ADDITIONAL SUCCESS: 2   # Diminishing returns
ACCURACY: 0.81          # Final performance: 81%
-----
```

### Primary Performance Metric

**Final Accuracy** after all trials:
```python
final_accuracy = num_successful_environments / total_environments
```

For the standard setup:
- 134 total environments
- 10 trials
- Final accuracy typically 0.70-0.85 (depending on model and memory usage)

---

## Analysis and Evaluation

### Experimental Comparison

The codebase supports comparing two conditions:

**Baseline (No Memory):**
```bash
./run_simple.sh
# Runs: python main.py --num_trials 10 --num_envs 134 --run_name "base_run_logs_gpt_35_turbo" --model "gpt-3.5-turbo"
# NO --use_memory flag
# Agent has NO access to reflections from past trials
# Each trial is independent
```

**Reflexion (With Memory):**
```bash
./run_reflexion.sh
# Runs: python main.py --num_trials 10 --num_envs 134 --run_name "reflexion_run_logs" --use_memory --model "gpt-3.5-turbo"
# --use_memory flag SET
# Agent accumulates reflections and learns from failures
```

### Analyzing Results from Logs

#### From world.log

Extract learning curves:
```bash
grep "ACCURACY" reflexion_run_logs/world.log
# Output:
# ACCURACY: 0.34
# ACCURACY: 0.52
# ACCURACY: 0.63
# ACCURACY: 0.71
# ACCURACY: 0.75
# ACCURACY: 0.78
# ACCURACY: 0.79
# ACCURACY: 0.80
# ACCURACY: 0.80
# ACCURACY: 0.81

grep "ADDITIONAL SUCCESS" reflexion_run_logs/world.log
# Output shows how many new tasks solved per trial
```

#### From JSON Files

**env_results_trial_9.json** (final trial state):
```json
[
  {
    "name": "env_0",
    "memory": [
      "I was stuck examining stoveburner repeatedly...",
      "I should check fridge before cabinets..."
    ],
    "is_success": true,
    "skip": false
  },
  {
    "name": "env_1",
    "memory": [],
    "is_success": true,
    "skip": false
  },
  {
    "name": "env_2",
    "memory": [
      "I could not find the apple in the usual locations...",
      "I should systematically check all containers...",
      "I need to open closed cabinets before assuming empty..."
    ],
    "is_success": false,
    "skip": false
  }
]
```

**Analysis:**
```python
import json

# Count final successes
with open('reflexion_run_logs/env_results_trial_9.json') as f:
    results = json.load(f)

successes = sum(1 for env in results if env['is_success'])
failures = sum(1 for env in results if not env['is_success'])
final_accuracy = successes / len(results)

print(f"Final accuracy: {final_accuracy:.2%}")
print(f"Successes: {successes}, Failures: {failures}")

# Analyze memory usage
avg_reflections = sum(len(env['memory']) for env in results) / len(results)
print(f"Average reflections per environment: {avg_reflections:.2f}")
```

### Advanced Analysis Techniques

#### 1. Learning Curve Comparison

```python
import matplotlib.pyplot as plt

baseline_accuracies = [0.34, 0.36, 0.35, 0.37, 0.36, 0.35, 0.37, 0.36, 0.35, 0.36]
reflexion_accuracies = [0.34, 0.52, 0.63, 0.71, 0.75, 0.78, 0.79, 0.80, 0.80, 0.81]

plt.figure(figsize=(10, 6))
plt.plot(range(10), baseline_accuracies, 'o-', label='Baseline (no memory)', linewidth=2)
plt.plot(range(10), reflexion_accuracies, 's-', label='Reflexion (with memory)', linewidth=2)
plt.xlabel('Trial Number', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.title('AlfWorld: Reflexion vs Baseline Learning Curves', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
```

#### 2. Task-Type Performance Analysis

```python
# Parse env names from JSON to determine task type
task_types = {
    'pick_and_place': [],
    'pick_clean_then_place': [],
    'pick_heat_then_place': [],
    'pick_cool_then_place': [],
    'look_at_obj': [],
    'pick_two_obj': []
}

# Group by task type and compute success rates
for task_type, envs in task_types.items():
    success_rate = sum(1 for e in envs if e['is_success']) / len(envs)
    print(f"{task_type}: {success_rate:.2%}")
```

#### 3. Reflection Effectiveness

```python
# Compare success rate vs number of reflections accumulated
reflection_counts = {}
for env in results:
    count = len(env['memory'])
    if count not in reflection_counts:
        reflection_counts[count] = {'success': 0, 'total': 0}
    reflection_counts[count]['total'] += 1
    if env['is_success']:
        reflection_counts[count]['success'] += 1

for count, stats in sorted(reflection_counts.items()):
    rate = stats['success'] / stats['total']
    print(f"{count} reflections: {rate:.2%} success ({stats['success']}/{stats['total']})")

# Example output:
# 0 reflections: 45% success (45/100)  # First-try successes
# 1 reflections: 68% success (15/22)   # Improved with 1 reflection
# 2 reflections: 75% success (6/8)     # Better with 2
# 3 reflections: 50% success (2/4)     # Hardest tasks, diminishing returns
```

#### 4. Failure Analysis

```python
# Examine remaining failures after all trials
failed_envs = [env for env in results if not env['is_success']]

print(f"Total unsolved: {len(failed_envs)}/134")
print(f"\nExample failed environment:")
print(f"Name: {failed_envs[0]['name']}")
print(f"Reflections accumulated: {len(failed_envs[0]['memory'])}")
print(f"Last reflection: {failed_envs[0]['memory'][-1]}")

# Common failure patterns:
# - Repeated reflections about same issue (not learning)
# - Tasks requiring complex multi-step reasoning
# - Object not found despite exhaustive search
# - Agent stuck in similar loops despite reflections
```

### Pre-Existing Results

The codebase includes logged results from previous runs:

```bash
# Baseline run results
ls root/base_run_logs/
# env_results_trial_0.json through env_results_trial_6.json

# Reflexion run results
ls root/reflexion_run_logs/
# env_results_trial_0.json through env_results_trial_14.json
```

You can parse these to reproduce paper results without running expensive LLM calls.

### Expected Paper-Style Results

#### Table: Final Performance Comparison

| Method | AlfWorld Success Rate | Trials to 75% |
|--------|---------------------|---------------|
| ReAct (no memory) | 34-36% | Never |
| ReAct + Last Trajectory | 52% | Never |
| **Reflexion (ours)** | **78-81%** | **5-6** |

#### Metrics Typically Reported

1. **Final Success Rate** - Primary metric
2. **Sample Efficiency** - Trials needed to reach X% success
3. **Per-Task-Type Performance** - Which tasks benefit most from reflection
4. **Ablation Studies** - Effect of memory size (1 vs 3 vs 5 reflections)

---

## Potential Improvements

Based on the analysis above, here are research directions to explore:

### 1. Improve Reflection Quality

**Current Limitation:** Single free-form reflection per failure

**Proposed:** Structured reflection with multiple aspects
```python
def generate_structured_reflection(log_str: str) -> Dict:
    return {
        'failed_action_pattern': "Repeated 'examine' action 4 times",
        'root_cause': "Did not use 'heat X with Y' command",
        'missing_exploration': "Never checked stoveburner 2, 3, 4",
        'strategy_adjustment': "Use heating command instead of examine",
        'priority_locations': ['stoveburner 2', 'microwave 1']
    }
```

**Benefits:**
- More specific, actionable guidance
- Easier for Actor to parse and use
- Can weight different reflection types

### 2. Better Loop Detection

**Current Limitation** (env_history.py:18-21): Only detects exact action repetition
```python
if value == self._last_action:
    self._is_exhausted = True
```

**Proposed:** Detect semantic loops
```python
def check_is_exhausted(self) -> bool:
    # Detect repeated action
    if self._last_action == self._history[-1]['value']:
        return True

    # Detect cycling through small set of actions
    if len(self._history) > 10:
        recent_actions = [h['value'] for h in self._history[-10:]]
        if len(set(recent_actions)) <= 3:
            return True  # Cycling through ≤3 actions

    # Detect location loops
    recent_locations = extract_locations(self._history[-8:])
    if len(set(recent_locations)) <= 2:
        return True  # Visiting same 2 locations repeatedly

    return False
```

### 3. Hierarchical Memory

**Current Limitation:** Flat list of last 3 reflections

**Proposed:** Short-term + long-term memory
```python
{
    'short_term': [last_3_reflections],  # Recent tactical failures
    'long_term': {                        # Compressed strategic learnings
        'never_do': [
            "Never examine same location >2 times",
            "Never repeat failed action without changing approach"
        ],
        'always_do': [
            "Check fridge first for food items",
            "Open closed containers before assuming empty"
        ],
        'task_strategies': {
            'heat': "Use microwave, not stoveburner examination",
            'clean': "Item in fridge/garbage, clean at sinkbasin"
        }
    }
}
```

### 4. Meta-Learning Across Task Types

**Current Limitation:** Each environment learns independently

**Proposed:** Share insights across similar tasks
```python
# After each trial, extract task-type patterns
task_type_learnings = {
    'pick_and_place': {
        'success_patterns': ["Check cabinets/drawers before countertops"],
        'failure_patterns': ["Examining same location repeatedly"]
    },
    'clean': {
        'success_patterns': ["Items in fridge/garbage, always use sinkbasin"],
        'failure_patterns': ["Forgetting to clean before placing"]
    }
}

# Inject task-specific hints into base prompt
base_prompt += f"\n\nTip for {task_type} tasks: {task_type_learnings[task_type]}"
```

### 5. Adaptive Memory Window

**Current Limitation:** Fixed 3 reflections regardless of task difficulty

**Proposed:** Dynamic memory size
```python
def get_memory_window(env_config, trial_num):
    num_reflections = len(env_config['memory'])

    # Increase memory for persistent failures
    if num_reflections >= 5:
        return env_config['memory'][-5:]  # Use more context
    elif num_reflections >= 3:
        return env_config['memory'][-3:]
    else:
        return env_config['memory']
```

### 6. Reflection Deduplication

**Current Limitation:** Can generate similar reflections multiple times

**Proposed:** Semantic deduplication with embeddings
```python
def add_reflection_if_novel(reflection: str, memory: List[str]) -> List[str]:
    # Embed new reflection
    new_emb = get_embedding(reflection)

    # Check similarity to existing reflections
    for old_reflection in memory:
        old_emb = get_embedding(old_reflection)
        similarity = cosine_similarity(new_emb, old_emb)

        if similarity > 0.85:  # Very similar
            return memory  # Don't add duplicate

    # Novel reflection, add it
    return memory + [reflection]
```

### 7. Counterfactual Reasoning

**Proposed:** Generate "what-if" scenarios in reflection
```python
reflection_query += """
Additionally, identify the critical decision point where you failed:
- What action did you take?
- What should you have done instead?
- How would the outcome differ?

Format:
CRITICAL FAILURE: At step X, I did Y
SHOULD HAVE DONE: Z
EXPECTED OUTCOME: Would have reached goal in N more steps
"""
```

### 8. Vector Database Integration (for large-scale)

**When it makes sense:**
- 100+ trials (long-term learning)
- Cross-domain transfer (multiple environments)
- Multi-agent systems sharing reflections

**Implementation:**
```python
# Store reflections with embeddings
vector_db.add(
    id=f"{env_name}_{trial}",
    embedding=get_embedding(reflection),
    metadata={
        'task_type': 'pick_and_place',
        'success': False,
        'trial': trial_idx
    },
    text=reflection
)

# Retrieve relevant reflections
def get_relevant_memory(current_task, k=3):
    query_emb = get_embedding(current_task)
    results = vector_db.query(query_emb, k=k, filter={'task_type': task_type})
    return [r.text for r in results]
```

---

## Summary

### Component Mapping

| Paper Component | Implementation | File Location |
|----------------|----------------|---------------|
| **Actor** (Ma) | `llm(env_history + ">")` | alfworld_trial.py:57 |
| **Evaluator** | `done` flag + `check_is_exhausted()` | alfworld_trial.py:67-70<br>env_history.py:23 |
| **Self-Reflection** (Msr) | `get_completion(reflection_query)` | generate_reflections.py:44 |
| **Memory** | `env_configs[i]['memory']` list | main.py:51<br>Updated at main.py:104 |

### Performance Metrics

| Metric | Calculation | Location |
|--------|-------------|----------|
| **Success Rate** | num_successes / total_envs | alfworld_trial.py:154 |
| **Additional Success** | New solves this trial | alfworld_trial.py:132 |
| **Learning Curve** | Track accuracy across trials | Extracted from world.log |
| **Final Performance** | Accuracy at trial N | Last env_results_trial_N.json |

### Key Files

- `main.py` - Master loop orchestrating trials
- `alfworld_trial.py` - Actor + Evaluator logic
- `generate_reflections.py` - Self-Reflection generation
- `env_history.py` - State tracking and loop detection
- `utils.py` - LLM API wrappers
- `prompts/alfworld_3prompts.json` - Few-shot examples for Actor
- `reflexion_few_shot_examples.txt` - Few-shot examples for Self-Reflection
