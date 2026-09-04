import os
import json
import openai
import time
import re

# ================= Security Configuration =================
API_KEY = "YOUR_API_KEY"          # Replace with your valid key
BASE_URL = "YOUR_BASE_URL"
MODEL_NAME = ""                   # 🔴 Please set your model name here (e.g., "gemini-3-flash-preview")

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= Global Background (Customised) =================
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

# ================= Core: LLM Semantic Fusion Prompt =================
FUSION_PROMPT = """
<Current_Task>Step_5_Final_Ontology_Fusion</Current_Task>
<Reference_Standard>
{GLOBAL_OBJECTIVE}
</Reference_Standard>

<Context>
Currently processing layer: 【{target_layer}】
You are required to perform a 【deep semantic fusion and deduplication】 of three candidate attribute dictionaries produced by three rounds of K‑Fold experiments.
The goal is to synthesise a single final “unified visual ontology Schema” with the strongest generalisation ability and no blind spots.
</Context>

<Input_Data>
Below are the candidate ontologies produced by the three experimental rounds:
{candidate_ontologies_json}
</Input_Data>

<Fusion_Protocols>
Please strictly adhere to the following protocols:
1. Cross‑round Semantic Alignment: Identify attribute dimensions that describe the same visual facet (e.g., round_1's "body_covering" and round_2's "skin_texture"). Merge them into a single standard name that is most generalisable and biologically intuitive.
2. Descriptive Synthesis: Combine the most precise definitions from the three inputs to rewrite the merged dimension's `definition`.
3. 🚨 Strict Evidence Retention 🚨: 【Crucial】 When merging multiple dimensions, you MUST perform a lossless union and deduplication of their original `source_cqs` (competency questions) and `visual_anchors`! Never lose the underlying visual evidence sources.
4. Orthogonality Constraint: Ensure that all final dimensions have no semantic overlap with each other.
</Fusion_Protocols>

<Requirement>
- If you performed any merging, renaming, or modification, set `has_changes` to true.
- If the current structure is already absolutely perfect, orthogonal, and has no evidence omissions, set `has_changes` to false.
</Requirement>

<Output_Format>
Must return standard JSON:
{{
    "has_changes": boolean,
    "fusion_log": [
        "Action: Merged Round 1's A and Round 2's B into C, and consolidated all their source_cqs.",
        "Action: Unified the definition for dimension D."
    ],
    "final_unified_ontology": {{
        "DimensionKey_1": {{
            "dimension_name": "Name",
            "definition": "Merged definition",
            "visual_anchors": ["merged anchor1", "merged anchor2"],
            "source_cqs": ["merged question1", "merged question2"]
        }}
    }}
}}
</Output_Format>
"""


def clean_json_response(content):
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def run_layer_fusion(base_dir="../outputs", target_layer="L1"):
    layer_dir = os.path.join(base_dir, target_layer)
    if not os.path.exists(layer_dir):
        print(f"⏭️ Skipping {target_layer}: directory does not exist")
        return

    # 1. Collect dictionaries from three rounds
    rounds = ["round_1", "round_2", "round_3"]
    candidates = {}

    for round_name in rounds:
        file_path = os.path.join(layer_dir, round_name, "step3_refined_ontology.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                if target_layer in data:
                    candidates[round_name] = data[target_layer]

    if not candidates:
        print(f"⚠️ [{target_layer}]: No dictionary files found for fusion.")
        return

    print(f"\n🚀 --- Starting LLM semantic fusion for layer {target_layer} ---")

    iteration_count = 0
    max_iters = 3
    is_stable = False

    current_input_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    final_ontology = {}
    all_fusion_history = []

    # 2. Meta‑cognitive self‑iteration loop
    while not is_stable and iteration_count < max_iters:
        iteration_count += 1
        print(f"  🔄 [Iteration {iteration_count}] AI refining fusion...")

        prompt = FUSION_PROMPT.format(
            GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
            target_layer=target_layer,
            candidate_ontologies_json=current_input_json
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                timeout=180
            )

            raw_content = clean_json_response(response.choices[0].message.content)
            res_data = json.loads(raw_content)

            has_changes = res_data.get("has_changes", True)
            fusion_log = res_data.get("fusion_log", [])
            final_ontology = res_data.get("final_unified_ontology", {})

            # Record history
            all_fusion_history.append({
                "iteration": iteration_count,
                "actions": fusion_log
            })

            for log in fusion_log:
                print(f"    🛠️ {log}")

            if not has_changes:
                print(f"  ✅ [{target_layer} ontology accepted]: Model confirms structure is optimal and logically closed.")
                is_stable = True
            else:
                # If changes remain, feed the current result back as input for the next iteration
                current_input_json = json.dumps({"current_merged_state": final_ontology}, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"  ❌ API or parsing error: {e}")
            time.sleep(5)
            break

    if not is_stable:
        print(f"  ⚠️ Reached maximum iterations ({max_iters}). Forcing fusion termination.")

    # 3. Wrap into the same nested format and save
    final_output_data = {target_layer: final_ontology}
    output_file = os.path.join(layer_dir, f"final_merged_ontology_{target_layer}.json")
    history_file = os.path.join(layer_dir, f"fusion_action_history_{target_layer}.json")

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(final_output_data, f, indent=4, ensure_ascii=False)

    with open(history_file, "w", encoding='utf-8') as f:
        json.dump(all_fusion_history, f, indent=4, ensure_ascii=False)

    print(f"  🏆 [{target_layer} fusion complete]: Produced {len(final_ontology)} unified dimensions!")
    print(f"     => Dictionary saved to: {output_file}")


if __name__ == "__main__":
    print("🌟 Starting K‑Fold Unified Visual Ontology Building Engine 🌟\n" + "=" * 50)
    # Iterate over your 6 layers and perform deep fusion one by one
    layers_to_process = ["L3", "L4", "L5", "L6"]

    for layer in layers_to_process:
        run_layer_fusion(base_dir="../outputs", target_layer=layer)

    print("\n🎉 All layers fused! Now you can proceed to the real‑image Grounding test phase!")