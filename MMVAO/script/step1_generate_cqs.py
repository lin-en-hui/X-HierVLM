import os
import json
import openai
import time

# Placeholder – replace with your own credentials
API_KEY = "YOUR_API_KEY"
BASE_URL = "YOUR_BASE_URL"
MODEL_NAME = "gemini-3-flash-preview"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)


def get_json_from_llm(prompt, retries=3):
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, temperature=1.0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ API request error (retry {i + 1}/{retries}): {e}")
            time.sleep(10)
    return {}


GLOBAL_OBJECTIVE = """
<Role>Expert in Biological Morphology and CV Ontology Architecture</Role>
<Objective>
Construct a "pure‑visual ontology for biological classification" that satisfies four iron rules:
1. Hierarchical alignment: strictly follows Linnaean taxonomy; high‑level taxa use macroscopic dimensions, low‑level taxa use detailed dimensions.
2. Purely visual: based only on features visible in a single static image; exclude functional, ecological, or anatomical attributes.
3. Orthogonal completeness: dimensions are non‑redundant and capable of describing any visible animal.
4. Zero‑shot generalisation: robust under adversarial stress tests.
</Objective>
"""

CQ_GENERATION_PROMPT = """
<Task>Step_1_CQ_Generation (Batch Mode)</Task>
<Reference>{GLOBAL_OBJECTIVE}</Reference>
<Context>
You will receive a JSON array containing multiple independent classification splits. For each `split_id`, propose a set of orthogonal "Competency Questions (CQs)".
Rules:
1. Purely visual: prohibit internal anatomy, genetics, or behavioural traits.
2. Combinatorial discriminative power: the combination of questions under the same split must yield a unique answer vector for every child category in that split.
3. Strictly restrict to visually perceptible dimensions.
4. Occam's razor: use as few questions as possible to achieve complete discrimination.
</Context>
<Input_Batch>
{batch_data}
</Input_Batch>
<Output_Format>
Return standard JSON:
{{
    "results": [
        {{
            "split_id": "must match input",
            "generated_cqs": [
                {{"cq": "A specific visual question?", "strategic_goal": "Brief reason"}}
            ]
        }}
    ]
}}
</Output_Format>
"""

CQ_JUDGE_PROMPT = """
<Task>Step_1_CQ_Audit (Batch Mode)</Task>
<Reference>{GLOBAL_OBJECTIVE}</Reference>
<Context>
You are now an auditor. You will receive a batch of Splits and their corresponding CQs.
For each Split, mentally construct a feature truth table and assess whether the CQ combination is sufficient to visually distinguish all its `children`.
1. If any non‑visual feature is present -> reject (false)
2. If two different children yield identical visual answers for all CQs -> reject (false)
3. Tolerate reasonable intra‑class variance.
</Context>
<Input_Batch>
{batch_data}
</Input_Batch>
<Output_Format>
Return standard JSON. 🚨ATTENTION🚨: if `is_perfect` is true, `rationale` must be exactly "OK", no extra text!
{{
    "results": [
        {{
            "split_id": "must match input",
            "is_perfect": boolean,
            "rationale": "OK (if true) OR specific failure reason (if false)"
        }}
    ]
}}
</Output_Format>
"""


class SplitTask:
    def __init__(self, split_dict):
        self.split_id = split_dict["split_id"]
        self.target_layer = split_dict["target_layer"]
        self.parent_node = split_dict["parent_node"]
        self.children = split_dict["children"]
        self.iteration_count = 0
        self.logs = []
        self.current_cqs = []


def run_step1(config_file="../config/auto_kfold_experiment_config_L3.json", round_idx=1):
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        round_key = f"round_{round_idx}"
        splits = config[round_key]["train_splits"]
        experiment_target_layer = config.get("target_layer", "L3")
    except Exception as e:
        print(f"⚠️ Failed to read config file: {e}")
        return

    # Isolate output folder by layer
    output_dir = f"../outputs/{experiment_target_layer}/{round_key}"
    os.makedirs(output_dir, exist_ok=True)

    final_output_path = os.path.join(output_dir, "step1_initial_cqs.json")
    history_output_path = os.path.join(output_dir, "step1_evolution_history.json")

    final_all_cqs = []
    full_evolution_history = []
    processed_split_ids = set()

    if os.path.exists(history_output_path) and os.path.exists(final_output_path):
        try:
            with open(final_output_path, "r", encoding='utf-8') as f:
                final_all_cqs = json.load(f).get("generated_cqs", [])
            with open(history_output_path, "r", encoding='utf-8') as f:
                history_data = json.load(f)
                for item in history_data.get("splits_history", []):
                    processed_split_ids.add(item["split_id"])
                    full_evolution_history.append(item)
            print(f"🔍 Resuming from checkpoint! Skipping {len(processed_split_ids)} already completed splits.")
        except:
            pass

    pending_queue = []
    for split in splits:
        if split["target_layer"] == experiment_target_layer and split["split_id"] not in processed_split_ids:
            pending_queue.append(SplitTask(split))

    start_time = time.time()
    MAX_CHILDREN_PER_BATCH = 20
    MAX_ITERATIONS = 5

    print(f"\n🚀 Starting batch processing pipeline for layer {experiment_target_layer}! (pending: {len(pending_queue)} splits)")

    while pending_queue:
        current_batch = []
        children_count = 0
        temp_queue = []

        while pending_queue:
            task = pending_queue.pop(0)
            c_len = len(task.children)
            if not current_batch or (children_count + c_len <= MAX_CHILDREN_PER_BATCH):
                current_batch.append(task)
                children_count += c_len
            else:
                temp_queue.append(task)

        pending_queue = temp_queue + pending_queue

        print(f"\n📦 [Batching]: {len(current_batch)} splits, {children_count} children in this batch")

        gen_inputs = []
        for task in current_batch:
            task.iteration_count += 1
            history_ctx = "First attempt." if not task.logs else f"Iter {task.logs[-1]['iter']} failed. Feedback: {task.logs[-1]['audit']}"
            gen_inputs.append({
                "split_id": task.split_id, "parent_node": task.parent_node,
                "children": task.children, "history_context": history_ctx
            })

        print("  🤖 [Generator]: Generating CQs in batch...")
        gen_res = get_json_from_llm(CQ_GENERATION_PROMPT.format(GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
                                                                batch_data=json.dumps(gen_inputs, ensure_ascii=False)))
        gen_map = {item.get("split_id"): item.get("generated_cqs", []) for item in gen_res.get("results", [])}

        judge_inputs = []
        for task in current_batch:
            task.current_cqs = gen_map.get(task.split_id, [])
            judge_inputs.append({"split_id": task.split_id, "parent_node": task.parent_node, "children": task.children,
                                 "current_cqs": task.current_cqs})

        print("  ⚖️  [Discriminator]: Auditing in batch...")
        judge_res = get_json_from_llm(CQ_JUDGE_PROMPT.format(GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
                                                             batch_data=json.dumps(judge_inputs, ensure_ascii=False)))
        judge_map = {item.get("split_id"): item for item in judge_res.get("results", [])}

        for task in current_batch:
            j_data = judge_map.get(task.split_id, {})
            is_perfect = j_data.get("is_perfect", False)
            rationale = j_data.get("rationale", "No feedback")

            task.logs.append(
                {"iter": task.iteration_count, "cqs": task.current_cqs, "is_perfect": is_perfect, "audit": rationale})

            if is_perfect or task.iteration_count >= MAX_ITERATIONS:
                for cq in task.current_cqs:
                    cq["split_id"] = task.split_id
                    cq["target_layer"] = task.target_layer
                    cq["parent_node"] = task.parent_node

                final_all_cqs.extend(task.current_cqs)
                full_evolution_history.append(
                    {"split_id": task.split_id, "status": "Perfect" if is_perfect else "Max_Limit",
                     "evolution_logs": task.logs})
                processed_split_ids.add(task.split_id)
                print(f"    {'✅ Passed' if is_perfect else '⚠️ Forced convergence'} | {task.split_id}")
            else:
                print(f"    ❌ Rejected | {task.split_id} -> Rationale: {rationale[:30]}...")
                pending_queue.append(task)

        with open(final_output_path, "w", encoding='utf-8') as f:
            json.dump({"generated_cqs": final_all_cqs}, f, indent=4, ensure_ascii=False)
        with open(history_output_path, "w", encoding='utf-8') as f:
            json.dump({"splits_history": full_evolution_history}, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 Done! Time elapsed: {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    run_step1()