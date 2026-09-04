import os
import json
import openai
import time

# ================= API Configuration =================
API_KEY = "YOUR_API_KEY"          # Replace with your valid key
BASE_URL = "YOUR_BASE_URL"
MODEL_NAME = "gemini-3-flash-preview"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)


def get_json_from_llm(prompt, retries=3):
    print("    🧠 [gemini-3-flash-preview] is silently reviewing and orthogonalising dimensions, please wait...")
    time.sleep(5)
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                timeout=240  # 🚨 Prevent network deadlock
            )
            raw_content = response.choices[0].message.content.strip()

            # Safely strip markdown formatting
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]

            return json.loads(raw_content.strip())
        except Exception as e:
            print(f"⚠️ API request error (retry {i + 1}/{retries}): {e}")
            time.sleep(10)
    return {}


# ================= Global Objective =================
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

# ================= Core: Meta‑cognitive Refinement Prompt (🟢 Compact version) =================
STEP3_REFINEMENT_PROMPT = """
<Task>Step_3_Ontology_Refinement</Task>
<Reference>{GLOBAL_OBJECTIVE}</Reference>

<Context>
Current classification layer: 【{target_layer}】 (total {dim_count} candidate dimensions).
Your task is to issue concrete modification instructions (Deletions / Merges) to refine the dictionary. Any dimension not mentioned will remain unchanged.
🎯 Target granularity: CV recognition typically benefits from 15‑30 highly discriminative dimensions. Your goal is to "eliminate synonym redundancy", not to "erase visual detail".
</Context>

<Input>
{current_ontology_json}
</Input>

<Protocols>
1. Redundancy Merges: Merge synonymous dimensions. 🚨【Anatomical isolation】: Absolutely forbid merging across different body parts (e.g., do not merge "shell" with "tentacle")!
2. Abstraction Corrections (Deletions): Delete invisible ecological/behavioural attributes. 🚨【Visual whitelist】: Colour, Pattern, Texture, Organ Morphology are strictly protected – never delete them!
3. Orthogonalisation: Ensure each dimension describes an independent physical facet. If Protocol 1 tries to merge "colour" with "morphology", veto it!
</Protocols>

<Output_Format>
Strictly output standard JSON.
🚨 Extremely important 🚨: As a strong reasoning model, perform all auditing logic internally. Do NOT output an `audit_log` or any analysis bloat in the JSON.
If the current system is already perfect (no redundancy, no invisible attributes), set `has_changes` to false and leave the rest as empty lists.
{{
    "has_changes": boolean,
    "deletions": ["key_to_delete_1", "key_to_delete_2"],
    "merges": [
        {{
            "source_dimensions": ["old_dim_key1", "old_dim_key2"],
            "target_dimension_name": "new_merged_dimension_name",
            "merged_definition": "New combined physical visual definition",
            "merged_visual_anchors": ["anchor1", "anchor2"]
        }}
    ]
}}
</Output_Format>
"""


# ================= [Core Execution Logic] =================
def run_step3(config_file="../config/auto_kfold_experiment_config_L3.json", round_idx=1):
    # 1. Dynamically obtain the target layer
    try:
        with open(config_file, "r", encoding='utf-8') as f:
            config = json.load(f)
            experiment_target_layer = config.get("target_layer", "Unknown_Layer")
    except Exception as e:
        print(f"⚠️ Failed to read config file: {e}")
        return

    round_key = f"round_{round_idx}"
    OUTPUT_DIR = f"../outputs/{experiment_target_layer}/{round_key}"
    HISTORY_DIR = os.path.join(OUTPUT_DIR, "step3_history_logs")
    os.makedirs(HISTORY_DIR, exist_ok=True)

    input_file = os.path.join(OUTPUT_DIR, "step2_layer_synthesized_dimensions.json")
    final_output_path = os.path.join(OUTPUT_DIR, "step3_refined_ontology.json")
    history_output_path = os.path.join(HISTORY_DIR, "step3_evolution_history.json")

    try:
        with open(input_file, "r", encoding='utf-8') as f:
            layered_schema = json.load(f)
    except FileNotFoundError:
        print(f"❌ Input file not found: {input_file}. Please run Step 2 first.")
        return

    current_ontology_input = layered_schema.get(experiment_target_layer, {})
    if not current_ontology_input:
        print(f"⚠️ No data for {experiment_target_layer} found in Step 2 output. Skipping refinement.")
        return

    print(f"\n🚀 --- Step 3: Starting meta‑cognitive refinement for {experiment_target_layer} ({round_key}) ---")
    evolution_history = {}
    start_time = time.time()

    print("\n" + "━" * 70)
    print(f"📍 [Current layer]: {experiment_target_layer} | Initial dimension count: {len(current_ontology_input)}")

    iteration_count = 0
    max_iters = 5
    is_stable = False
    layer_audit_history = []

    while not is_stable and iteration_count < max_iters:
        iteration_count += 1
        print(f"  🔄 [Refinement iteration {iteration_count} in progress...]")

        # Strip source_cqs for LLM input to save tokens
        clean_input_for_llm = {}
        for k, v in current_ontology_input.items():
            clean_input_for_llm[k] = {
                "dimension_name": v.get("dimension_name", ""),
                "definition": v.get("definition", ""),
                "visual_anchors": v.get("visual_anchors", [])
            }

        prompt = STEP3_REFINEMENT_PROMPT.format(
            GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
            target_layer=experiment_target_layer,
            dim_count=len(clean_input_for_llm),
            current_ontology_json=json.dumps(clean_input_for_llm, ensure_ascii=False, indent=2)
        )

        try:
            res_data = get_json_from_llm(prompt)

            has_changes = res_data.get("has_changes", False)
            deletions = res_data.get("deletions", [])
            merges = res_data.get("merges", [])

            # Record compact history log
            layer_audit_history.append({
                "iteration": iteration_count,
                "changes_made": has_changes,
                "deletions_count": len(deletions),
                "merges_count": len(merges),
                "deletions_detail": deletions,
                "merges_detail": merges
            })

            # If perfectly converged
            if not has_changes and not deletions and not merges:
                print(f"    ✅ [Ontology stable]: o1 confirms that {experiment_target_layer} dimensions are absolutely orthogonal; no further changes needed!")
                is_stable = True
                break

            action_count = 0

            # 🛠️ Execute Deletions
            for key_to_delete in deletions:
                if key_to_delete in current_ontology_input:
                    del current_ontology_input[key_to_delete]
                    action_count += 1

            # 🛠️ Execute Merges
            for merge_cmd in merges:
                sources = merge_cmd.get("source_dimensions", [])
                new_name = merge_cmd.get("target_dimension_name", "")

                if not sources or not new_name:
                    continue

                combined_cqs = []
                valid_merge = False
                for src in sources:
                    if src in current_ontology_input:
                        combined_cqs.extend(current_ontology_input[src].get("source_cqs", []))
                        del current_ontology_input[src]
                        valid_merge = True

                if valid_merge:
                    current_ontology_input[new_name] = {
                        "dimension_name": new_name,
                        "definition": merge_cmd.get("merged_definition", ""),
                        "visual_anchors": merge_cmd.get("merged_visual_anchors", []),
                        "source_cqs": list(set(combined_cqs))
                    }
                    action_count += 1

            print(
                f"    🛠️  Python successfully executed {action_count} pruning actions (deletions: {len(deletions)}, merges: {len(merges)}).")

            # Defensive check
            if action_count == 0:
                print("    ⚠️ [Warning] Model claimed changes but provided no actionable operations. Forcing convergence.")
                is_stable = True

        except Exception as e:
            print(f"    ❌ [Review crash]: {e}")
            break

    if not is_stable:
        print(f"  🛑 [Warning]: Reached maximum iterations ({max_iters}). Forcing termination.")

    layered_schema[experiment_target_layer] = current_ontology_input
    evolution_history[experiment_target_layer] = {
        "total_iterations": iteration_count,
        "convergence_reached": is_stable,
        "audit_logs": layer_audit_history
    }

    print(f"  🏆 [{experiment_target_layer} refinement complete]: Dimension count safely optimised to {len(current_ontology_input)}!")

    with open(final_output_path, "w", encoding='utf-8') as f:
        json.dump(layered_schema, f, indent=4, ensure_ascii=False)

    with open(history_output_path, "w", encoding='utf-8') as f:
        json.dump(evolution_history, f, indent=4, ensure_ascii=False)

    total_time = (time.time() - start_time) / 60
    print("\n" + "🎉" * 5 + f"  Done! Ontology cleaning and orthogonalisation for {experiment_target_layer} ({round_key}) complete!" + "🎉" * 5)
    print(f"🕰️  Run time: {total_time:.1f} minutes. Dictionary saved to: {final_output_path}")


if __name__ == "__main__":
    run_step3(config_file="../config/auto_kfold_experiment_config_L5.json", round_idx=2)