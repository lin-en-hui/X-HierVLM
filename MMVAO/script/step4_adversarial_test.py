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
    print("    🧠 [gemini-3-flash-preview blind test] is performing high‑intensity 'Lego‑like deduction' in the background, please wait...")
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                timeout=240
            )
            raw_content = response.choices[0].message.content.strip()

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
1. Hierarchical alignment: strictly follows Linnaean taxonomy (L1–L6); high‑level taxa use macroscopic dimensions, low‑level taxa use detailed dimensions.
2. Purely visual: based only on features visible in a single static image; exclude functional, ecological, or invisible attributes.
3. Orthogonal completeness: dimensions are non‑redundant (No Superfluous) and capable of describing any visible animal.
4. Zero‑shot generalisation: robust under adversarial stress tests.
</Objective>
"""

# ================= Step 4: Coverage test prompt =================
STEP4_COVERAGE_PROMPT = """
<Current Task: Step_4_Taxonomy_Coverage_Test>
<Reference_Standard>
{GLOBAL_OBJECTIVE}
</Reference_Standard>

<Context>
You are now required to perform an 【Attribute‑Dimension Coverage Test】.
I will provide a list of specific taxonomic animal names (the current test layer is {layer_description}).
You must rigorously review the current attribute dictionary and judge whether it is sufficient to **completely and accurately describe** the core visual features of these animals.
</Context>

<Input Data>
1. The attribute‑dimension ontology Schema to be tested (current layer only):
{ontology_json}

2. The current test batch of taxa (Taxa List):
{taxa_list}
</Input Data>

<Execution_Protocol>
For each taxon in the list, strictly perform the following audit:
“Note: Your objective is to maintain the ontology’s ‘completeness, orthogonality, and generalisability’. Do not suggest adding a dimension that can only describe a specific taxon just to make that taxon PASS.”

【Highest Red Line / CRITICAL WARNING】:
You **must and can only** use the attribute dimensions (Keys) and their descriptions (Values) provided in the given JSON dictionary ({ontology_json}) to compose and describe the animals.
Absolutely prohibit using your prior biological knowledge! If a critical visual feature of an animal cannot be mapped to any dimension in the provided dictionary, you MUST immediately judge it as FAIL (collision_detected: true)!
If the provided dictionary is empty `{{}}`, you MUST unconditionally return all animals as FAIL!

1. **Compositional Attempt**:
   - Core task: try to compose the animal using multiple existing attribute dimensions in a “Lego‑like” fashion.
   - Judgement: if the combination “dimension A + dimension B + dimension C + ...” is sufficient to outline the core visual features and distinguish it from others, it is considered a 【successful description】.
   - Strictly forbidden: do NOT fail just because there is no “dedicated dimension” or “single matching term”.

2. **Semantic Boundary Check**:
   - Core task: distinguish between “missing value” and “missing dimension”.
   - Judgement: if the only missing thing is a specific descriptor (e.g., the “conical horn” of a particular rhino or the “hooked beak” of a specific bird), but these features could clearly be accommodated under existing dimensions like 【Horn_Morphology】 or 【Beak_Shape】, then it is a “value‑space to be expanded” and MUST be judged as **PASS**.

3. **True Blind Spot Judgement**:
   - Core task: identify core visual perception gaps.
   - Judgement: only when a core visual attribute of the object has no corresponding dimension in the existing system (e.g., completely unable to assign “scales” to any dimension) shall it be judged as **FAIL**.
</Execution_Protocol>

<Output_Format>
You possess a powerful implicit Chain‑of‑Thought. Perform all the above reconstruction attempts and blind‑spot analysis in your **internal reasoning**.
To save tokens and ensure stable JSON parsing, **do NOT include lengthy analysis in the final JSON output**!

Strictly output the following minimal JSON format:
{{
    "collision_detected": boolean,
    "verdict_reason": "One sentence (within 20 words) explaining the core reason for PASS or FAIL",
    "supplemental_cqs": [
        {{
            "failed_taxa": "Which specific taxon/taxa caused the failure",
            "cq": "Write a highly specific Competency Question (CQ) in English. This question MUST target the exact visual/morphological structures of this taxon that are CURRENTLY MISSING from the dictionary. Force the discovery of NEW dimensions (e.g., 'What are the unique visual characteristics of the specific feeding apparatus/defensive armor in [Taxon]?'). Do NOT ask about attributes already present in the existing dictionary."
        }}
    ]
}}
Note: if collision_detected is false, keep supplemental_cqs as an empty list [].
</Output_Format>
</Current Task>
"""


def run_step4_summary_mode(config_file="../config/auto_kfold_experiment_config_L5.json", round_idx=1, loop_idx=1,
                           batch_size=10):
    try:
        with open(config_file, "r", encoding='utf-8') as f:
            config = json.load(f)
            target_layer_prefix = config.get("target_layer", "L5")
            round_key = f"round_{round_idx}"
            taxonomy_summary = config[round_key]["adversarial_metrics"]["taxonomy_summary"]
    except Exception as e:
        print(f"❌ Failed to read config file: {e}")
        return

    OUTPUT_DIR = f"../outputs/{target_layer_prefix}/{round_key}"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ontology_file = os.path.join(OUTPUT_DIR, "step3_refined_ontology.json")
    report_output_file = os.path.join(OUTPUT_DIR, f"step4_coverage_report_loop{loop_idx}.json")
    step1_cq_file = os.path.join(OUTPUT_DIR, "step1_initial_cqs.json")

    # 💡 [New] Error‑book file path
    failed_taxa_file = os.path.join(OUTPUT_DIR, "step4_failed_taxa.json")

    try:
        with open(ontology_file, "r", encoding='utf-8') as f:
            current_ontology = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read ontology: {e}\nPlease ensure Step 3 has been completed.")
        return

    LAYER_MAPPING = {"L1": "L1_Phylums", "L2": "L2_Classes", "L3": "L3_Orders", "L4": "L4_Families", "L5": "L5_Genuses",
                     "L6": "L6_Species"}
    target_layer_key = LAYER_MAPPING.get(target_layer_prefix, "L3_Orders")

    # 💡 [Core modification]: if loop_idx == 1, load full data; if > 1, only load the error book
    if loop_idx == 1 or not os.path.exists(failed_taxa_file):
        taxa_list = taxonomy_summary.get(target_layer_key, [])
        print(f"\n✅ [Loop {loop_idx}] Initial full‑scale target practice: extracted {len(taxa_list)} test taxa in total!")
    else:
        with open(failed_taxa_file, "r", encoding='utf-8') as f:
            taxa_list = json.load(f)
        print(f"\n🎯 [Loop {loop_idx}] Error‑book sniping: only testing the {len(taxa_list)} previously failed taxa!")

    if not taxa_list:
        print("🎉 The test list is empty – the current layer has passed perfectly!")
        return

    print(f"🚀 --- Step 4: Starting coverage audit (exclusive for {target_layer_prefix}) ---")

    layer_ontology_dict = current_ontology.get(target_layer_prefix, {})
    start_time = time.time()

    # Record taxa that still fail this round
    next_loop_failed_taxa = []
    all_supplemental_cqs = []
    full_report = []

    for i in range(0, len(taxa_list), batch_size):
        batch_taxa = taxa_list[i: i + batch_size]
        batch_id = f"Batch_{i // batch_size + 1}"

        print(f"\n⚔️ [Testing {batch_id}]: {', '.join(batch_taxa)}")

        prompt = STEP4_COVERAGE_PROMPT.format(
            GLOBAL_OBJECTIVE=GLOBAL_OBJECTIVE,
            ontology_json=json.dumps(layer_ontology_dict, ensure_ascii=False),
            layer_description=target_layer_prefix,
            taxa_list=json.dumps(batch_taxa, ensure_ascii=False)
        )

        try:
            result = get_json_from_llm(prompt)
            collision_detected = result.get("collision_detected", False)
            verdict_reason = result.get("verdict_reason", "No clear reason")
            new_cqs = result.get("supplemental_cqs", [])

            full_report.append({
                "batch_id": batch_id,
                "tested_taxa": batch_taxa,
                "collision_detected": collision_detected,
                "verdict_reason": verdict_reason
            })

            if collision_detected:
                print(f"  ❌ [FAIL - Blind spot found]: {verdict_reason}")
                # 💡 [New] If this batch fails, push all its taxa into the error book for the next loop
                next_loop_failed_taxa.extend(batch_taxa)

                for cq in new_cqs:
                    cq["source_tag"] = f"Step4_Coverage_{target_layer_prefix}_Loop_{loop_idx}"
                    cq["batch_id"] = batch_id
                    cq["target_layer"] = target_layer_prefix
                all_supplemental_cqs.extend(new_cqs)
            else:
                print(f"  ✅ [PASS - Description successful]: The current dictionary sufficiently covers this batch.")

        except Exception as e:
            print(f"  ⚠️ Error while testing batch {batch_id}: {e}")
            next_loop_failed_taxa.extend(batch_taxa)  # even on error, treat as failed and retry next loop

    with open(report_output_file, "w", encoding='utf-8') as f:
        json.dump(full_report, f, indent=4, ensure_ascii=False)

    # 💡 [New] Overwrite the error book with the newly failed taxa
    with open(failed_taxa_file, "w", encoding='utf-8') as f:
        json.dump(next_loop_failed_taxa, f, indent=4, ensure_ascii=False)

    if all_supplemental_cqs:
        print("\n" + "⚠️" * 5)
        print(f"Audit complete! {len(next_loop_failed_taxa)} taxa have been added to the next loop's error book.")
        print(f"Automatically generated {len(all_supplemental_cqs)} supplemental CQs.")

        if os.path.exists(step1_cq_file):
            try:
                with open(step1_cq_file, "r", encoding='utf-8') as f:
                    s1_data = json.load(f)
                s1_data.setdefault("generated_cqs", []).extend(all_supplemental_cqs)
                with open(step1_cq_file, "w", encoding='utf-8') as f:
                    json.dump(s1_data, f, indent=4, ensure_ascii=False)

                # 2. [New] Write the newly generated CQs to a dedicated file for Step 2 consumption
                loop_cqs_file = os.path.join(os.path.dirname(step1_cq_file), "current_loop_new_cqs.json")
                loop_data = {"generated_cqs": all_supplemental_cqs}
                with open(loop_cqs_file, "w", encoding='utf-8') as f:
                    json.dump(loop_data, f, indent=4, ensure_ascii=False)

                print(
                    f"🔄 [Closed‑loop sync]: New CQs appended! Please run Step 2 to extract features -> Step 2.5 merge -> Step 4 (Loop {loop_idx + 1}) to continue shooting the error book.")
            except Exception as e:
                pass
    else:
        print("\n🎉🎉🎉 Perfect pass! The error book is now empty! The current attribute dictionary perfectly covers all test taxa!")

    print(f"\n🕰️ Test time elapsed: {(time.time() - start_time) / 60:.1f} minutes.")


if __name__ == "__main__":
    run_step4_summary_mode(
        config_file="../config/auto_kfold_experiment_config_L5.json",
        round_idx=1,
        loop_idx=1,
        batch_size=10
    )