import os
import json
import openai
import time
import re

# ================= API Configuration =================
API_KEY = "YOUR_API_KEY"
BASE_URL = "YOUR_BASE_URL"
MODEL_NAME = "gemini-3-flash-preview"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)


def get_json_from_llm(prompt, retries=3):
    print("    🧠 [gemini-3-flash-preview] is performing high-intensity visual feature extraction, please wait...")
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                timeout=300  # Extended to 5 minutes to prevent disconnection
            )
            raw_content = response.choices[0].message.content.strip()

            # 🟢 [Defense Line 1]: Regex to extract the core JSON block, ignoring surrounding text and Markdown
            json_match = re.search(r'(\{.*\}|\[.*\])', raw_content, re.DOTALL)
            if json_match:
                clean_json_str = json_match.group(1)
            else:
                clean_json_str = raw_content

            return json.loads(clean_json_str)

        except json.JSONDecodeError as e:
            print(f"    ⚠️ JSON parsing failed, model returned garbled output (retry {i + 1}/{retries}): {e}")
            time.sleep(5)
        except Exception as e:
            print(f"    ⚠️ API request error (retry {i + 1}/{retries}): {e}")
            time.sleep(10)
    return {}


# ================= Global Objective =================
GLOBAL_OBJECTIVE = """
<Role>Expert in Biological Morphology and CV Ontology Architecture</Role>
<Objective>
Construct a "pure‑visual ontology for biological classification" that satisfies four iron rules:
1. Hierarchical alignment: strictly follows Linnaean taxonomy (L1–L6); high‑level taxa use macroscopic dimensions, low‑level taxa use detailed dimensions.
2. Purely visual: based only on features visible in a single static image; exclude functional, ecological, or invisible attributes.
3. Orthogonal completeness: dimensions are non‑redundant (No Superfluous) and capable of describing any visible animal.
4. Zero‑shot generalisation: robust under adversarial stress tests.
</Objective>
"""

# ================= Core: Incremental Extraction Prompt =================
STEP2_ITERATIVE_PROMPT = """
<Current Task: Step_2_Iterative_Dimension_Synthesis>
<Reference_Standard>{GLOBAL_OBJECTIVE}</Reference_Standard>

<Context>
You are a CV ontologist specialising in Open‑World Object Detection (OWOD). For the layer 【{target_layer}】 (containing categories: {layer_categories}), review the CQs below and decide whether the current attribute dictionary is complete.
If any dimension is missing, extract entirely new dimensions.
【Pixel‑level evidence principle】：Only extract dimensions that can be directly evidenced by anatomical or surface features in a single static frame. Reject ecological or behavioural attributes.
</Context>

<Input Data>
1. Unabsorbed incremental CQs (total {cq_count}):
{cqs_list}

2. Currently accumulated attribute dimensions:
{current_schema_context}
</Input Data>

<Execution_Protocol>
🚨【Highest Red Line: Anatomical Decoupling & Universality】:
Your dimension names MUST be generic body‑part or physical‑property terms. Absolutely prohibit category‑specific "catch‑all" names (e.g., "Bird Traits", "Mammal Body Plan", "Insect Morphology")!
Different body parts must be split into separate dimensions. Never combine "mouth shape", "toe structure", and "surface texture" into a single dimension!

1. **Generic Lego‑brick orientation**:
   - Extracted dimensions must be cross‑taxa universal (e.g., use `Mouthpart_Morphology` to describe bird beaks, fish snouts, insect mouthparts; use `Body_Covering` for feathers, scales, shells).
   - Even for future unknown chimeric species (e.g., platypus), they can be described via combinations of these generic anatomical dimensions.
2. **Pixel‑level evidence principle**: Ensure each extracted dimension is directly observable in a single static frame.
3. **Greedy Absorption**: Do not be lazy! If the provided CQs indeed contain multiple entirely new independent visual features, extract all corresponding new dimensions in one go – do not defer to later rounds.
4. **Review and absorb**: Compare the “accumulated dimensions” with the “incremental CQs”. If any visual feature is missing, output the new dimension in JSON, **and make sure to copy the original CQ text verbatim into `source_cqs`**.
5. If the existing system already covers everything perfectly, or if the remaining CQs are meaningless/invisible garbage questions, simply output an empty list `[]`.
</Execution_Protocol>

<Output_Format>
【🚨 Extremely strict output instruction 🚨】
You MUST output only a single valid JSON format!
Directly outputting a JSON array (List) is strictly forbidden! The outermost container must be a JSON object (Dictionary) with a single key (`{target_layer}`).

✅ Correct example:
{{
    "{target_layer}": [
        {{
            "dimension_name": "English dimension phrase (must be a body part or physical property, e.g., Snout_Morphology, Limb_Structure)",
            "definition": "Physical visual definition, clearly specifying which pixel‑level morphological features to look for",
            "visual_anchors": ["value_example_1", "value_example_2"],
            "source_cqs": ["Original CQ text 1 addressed by this dimension", "Original CQ text 2 addressed by this dimension"]
        }}
    ]
}}
</Output_Format>
</Current Task>
"""


def run_step2(config_file="../config/auto_kfold_experiment_config_L3.json", round_idx=1):
    round_key = f"round_{round_idx}"
    try:
        with open(config_file, "r", encoding='utf-8') as f:
            config = json.load(f)
            experiment_target_layer = config.get("target_layer", "Unknown_Layer")
            train_splits = config.get(round_key, {}).get("train_splits", [])
            layer_categories_set = set()
            for split in train_splits:
                if split.get("target_layer") == experiment_target_layer:
                    for child in split.get("children", []):
                        layer_categories_set.add(child)
            layer_cats_str = ", ".join(list(layer_categories_set))
            if len(layer_cats_str) > 300:
                layer_cats_str = layer_cats_str[:300] + "...(and more categories)"
    except Exception as e:
        print(f"⚠️ Failed to read config file: {e}")
        return

    OUTPUT_DIR = f"../outputs/{experiment_target_layer}/{round_key}"
    HISTORY_DIR = os.path.join(OUTPUT_DIR, "step2_history_logs")
    os.makedirs(HISTORY_DIR, exist_ok=True)

    # 👇 [Modified] Smart data source selection (cold start vs incremental update)
    step1_cq_file = os.path.join(OUTPUT_DIR, "step1_initial_cqs.json")
    loop_cqs_file = os.path.join(OUTPUT_DIR, "current_loop_new_cqs.json")

    if os.path.exists(loop_cqs_file):
        input_cq_file = loop_cqs_file
        print(f"    📥 [Data source] Detected incremental magazine, reading newly supplemented CQs ({loop_cqs_file})...")
    else:
        input_cq_file = step1_cq_file
        print(f"    📥 [Data source] Initial round, reading base question bank ({step1_cq_file})...")

    all_dims_file = os.path.join(OUTPUT_DIR, "step2_layer_synthesized_dimensions.json")
    discarded_cq_file = os.path.join(OUTPUT_DIR, "step2_discarded_cqs.json")

    # Load previously discarded CQs
    discarded_cqs = set()
    if os.path.exists(discarded_cq_file):
        try:
            with open(discarded_cq_file, "r", encoding='utf-8') as f:
                discarded_cqs = set(json.load(f))
        except Exception:
            pass

    raw_cqs = []
    try:
        with open(input_cq_file, "r", encoding='utf-8') as f:
            cqs_data = json.load(f)
            raw_cqs.extend(cqs_data.get("generated_cqs", []))
    except Exception as e:
        print(f"    ⚠️ Failed to read CQ data file: {e}")
        return

    layer_cqs = [item for item in raw_cqs if item.get("target_layer") == experiment_target_layer]
    if not layer_cqs:
        print(f"    ⚠️ No CQs found for layer {experiment_target_layer} in the current data source. Skipping.")
        return

    print(f"\n🚀 --- Step 2: Starting incremental dimension synthesis for layer {experiment_target_layer} ({round_key}) ---")

    # Use the cleaned step3 dictionary as prior knowledge for the LLM, not a dirty cache
    step3_refined_file = os.path.join(OUTPUT_DIR, "step3_refined_ontology.json")

    global_schema = {}
    current_layer_schema = {}  # This is the "clean dictionary" shown to the LLM
    if os.path.exists(step3_refined_file):
        try:
            with open(step3_refined_file, "r", encoding='utf-8') as f:
                global_schema = json.load(f)
                current_layer_schema = global_schema.get(experiment_target_layer, {})
        except Exception:
            pass

    # 👇 [New] Build a set of already‑existing source CQs globally
    existing_source_cqs = set()
    for dim_data in current_layer_schema.values():
        dim_cqs = dim_data.get("source_cqs", [])
        existing_source_cqs.update(dim_cqs)

    # This is the fresh cache for newly generated dimensions in this loop; initially empty
    newly_synthesized_this_loop = {}

    start_time = time.time()

    clean_cqs = []
    for item in layer_cqs:
        cq_text = item.get("cq", item.get("new_cq", ""))
        if cq_text and cq_text not in discarded_cqs:
            clean_cqs.append(cq_text)

    iteration = 0
    max_iterations = 6
    is_saturated = False

    while not is_saturated and iteration < max_iterations:
        existing_source_cqs = set()
        for dim_data in current_layer_schema.values():
            existing_source_cqs.update(dim_data.get("source_cqs", []))

        # 👇 [Modified] Triple filter: keep only CQs not in discarded list, and not already in the dictionary
        unprocessed_cqs = [
            cq for cq in clean_cqs
            if cq not in discarded_cqs and cq not in existing_source_cqs
        ]

        if not unprocessed_cqs:
            print(f"    ✅ [Converged]: All valid CQs have been absorbed into the dictionary. No further extraction needed.")
            break

        iteration += 1
        print("\n" + "━" * 50)
        print(f"  🔄 [Iteration {iteration}] Remaining CQs to process: {len(unprocessed_cqs)} / Initial total: {len(clean_cqs)}")

        schema_context_str = json.dumps(current_layer_schema, indent=2,
                                        ensure_ascii=False) if current_layer_schema else "None (Initial Empty State)"

        prompt = STEP2_ITERATIVE_PROMPT.format(
            GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
            target_layer=experiment_target_layer,
            layer_categories=layer_cats_str,
            cq_count=len(unprocessed_cqs),
            cqs_list=json.dumps(unprocessed_cqs, ensure_ascii=False),
            current_schema_context=schema_context_str
        )

        result = get_json_from_llm(prompt)

        new_dims_list = []
        if isinstance(result, list):
            new_dims_list = result
        elif isinstance(result, dict):
            if experiment_target_layer in result:
                new_dims_list = result[experiment_target_layer]
            else:
                keys = list(result.keys())
                if keys and isinstance(result[keys[0]], list):
                    new_dims_list = result[keys[0]]

        added_count = 0
        if new_dims_list:
            for dim in new_dims_list:
                dim_name = dim.get("dimension_name", "")
                if not dim_name:
                    continue

                # New dimension: update both the context and the clean cache
                if dim_name not in current_layer_schema:
                    current_layer_schema[dim_name] = dim  # update context for LLM
                    newly_synthesized_this_loop[dim_name] = dim  # 🌟 Core: only store brand‑new dimensions for later review
                    added_count += 1
                else:
                    existing_cqs = current_layer_schema[dim_name].get("source_cqs", [])
                    new_cqs = dim.get("source_cqs", [])
                    current_layer_schema[dim_name]["source_cqs"] = list(set(existing_cqs + new_cqs))

        if added_count == 0:
            print(f"    ✅ [Convergence achieved]: Model confirms no new dimensions can be extracted. Auto‑terminating mining!")

            if unprocessed_cqs:
                print(f"    🗑️ [Cost‑saving mode enabled]: Discarding remaining {len(unprocessed_cqs)} invalid CQs and adding them to the blacklist.")
                discarded_cqs.update(unprocessed_cqs)
                with open(discarded_cq_file, "w", encoding='utf-8') as f:
                    json.dump(list(discarded_cqs), f, indent=4, ensure_ascii=False)

            is_saturated = True
        else:
            print(f"    ⚠️ [Continuing mining]: Discovered {added_count} new visual dimensions. Moving to next round...")
            time.sleep(3)

        # Real‑time saving: only overwrite the cache file with the newly extracted dimensions for Step 2.5 review
        with open(all_dims_file, "w", encoding='utf-8') as f:
            json.dump({experiment_target_layer: newly_synthesized_this_loop}, f, indent=4, ensure_ascii=False)

    total_time = (time.time() - start_time) / 60
    print(
        f"\n🏆 [{experiment_target_layer} synthesis complete]: Dictionary now contains {len(current_layer_schema)} orthogonal dimensions! Time elapsed: {total_time:.1f} min")


if __name__ == "__main__":
    run_step2(config_file="../config/auto_kfold_experiment_config_L5.json", round_idx=1)