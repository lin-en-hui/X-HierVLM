import os
import json
import random

# ================= 1. Configuration =================
# 🔴 Please replace with your actual dataset directory path
DATA_DIR = "YOUR_DATASET_DIRECTORY_PATH"
# 🔴 Please replace with your desired output directory for config files
OUTPUT_DIR = "YOUR_OUTPUT_DIRECTORY_PATH"
# Default to generate 3-fold
NUM_FOLDS = 3


# ================= 2. Core Functions =================

def parse_directory_to_taxonomy(data_dir):
    """Automatically extract Linnaean taxonomy from folder names and build a full animal dictionary."""
    animal_taxonomy = {}

    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}. Please check the path.")
        return animal_taxonomy

    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            parts = folder_name.split('_')
            # Ensure standard format and belongs to Animalia: 00000_Animalia_Phylum_Class_Order_Family_Genus_Species
            if len(parts) >= 8 and parts[1] == 'Animalia':
                phylum = parts[2]
                class_name = parts[3]
                order = parts[4]
                family = parts[5]
                genus = parts[6]
                # Species name uses Linnaean binomial nomenclature (genus + specific epithet)
                species = " ".join(parts[6:])

                animal_taxonomy[species] = {
                    "L1": phylum,
                    "L2": class_name,
                    "L3": order,
                    "L4": family,
                    "L5": genus,
                    "L6": species
                }
    return animal_taxonomy


def extract_taxonomy_summary(animals_dict):
    """Dynamically extract statistical summaries for all layers (L1-L6)."""
    if not animals_dict:
        return {}

    layers = {
        "L1_Phylums": "L1",
        "L2_Classes": "L2",
        "L3_Orders": "L3",
        "L4_Families": "L4",
        "L5_Genuses": "L5",
        "L6_Species": "L6"
    }

    summary = {k: set() for k in layers.keys()}

    for info in animals_dict.values():
        for label, key in layers.items():
            if key in info:
                summary[label].add(info[key])

    return {k: sorted(list(v)) for k, v in summary.items()}


def extract_branching_splits(animals_dict):
    """Extract branching nodes from the flattened animal dictionary."""
    splits = []
    layers = ["L1", "L2", "L3", "L4", "L5", "L6"]

    # Extract L1 root nodes
    l1_nodes = set(info["L1"] for info in animals_dict.values())
    if len(l1_nodes) > 1:
        splits.append({
            "split_id": "split_L1_Root",
            "target_layer": "L1",
            "parent_node": "Root",
            "children": list(l1_nodes)
        })

    # Extract L2-L6 branching nodes
    for i in range(1, len(layers)):
        current_layer = layers[i]
        parent_layer = layers[i - 1]

        parent_to_children = {}
        for info in animals_dict.values():
            p = info[parent_layer]
            c = info[current_layer]
            if p not in parent_to_children:
                parent_to_children[p] = set()
            parent_to_children[p].add(c)

        for p, c_set in parent_to_children.items():
            if len(c_set) > 1:
                splits.append({
                    "split_id": f"split_{current_layer}_{p}",
                    "target_layer": current_layer,
                    "parent_node": p,
                    "children": list(c_set)
                })
    return splits


# ================= 3. Main Workflow =================

def main():
    print(f"🚀 Starting to parse dataset directory: {DATA_DIR} ...")
    global_taxonomy = parse_directory_to_taxonomy(DATA_DIR)

    total_species = len(global_taxonomy)
    if total_species == 0:
        return
    print(f"✅ Successfully extracted taxonomy information for {total_species} species!")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    layers = ["L1", "L2", "L3", "L4", "L5", "L6"]

    # Generate independent K-Fold configs for each layer from L1 to L6
    for target_layer in layers:
        print(f"\n" + "=" * 50)
        print(f"🎯 Generating K-Fold config for layer 【{target_layer}】...")

        # 1. Get all unique category names for the current layer (e.g., all "Orders" for L3)
        unique_categories = list(set([info[target_layer] for info in global_taxonomy.values()]))

        # Handle cases with too few categories (e.g., only one phylum)
        actual_k = min(NUM_FOLDS, len(unique_categories))
        if actual_k < 2:
            print(f"⚠️ Layer {target_layer} has only {len(unique_categories)} unique categories, which is insufficient for K-Fold. Skipped.")
            continue

        # 2. Shuffle and split the categories for the current layer
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(unique_categories)

        fold_size = len(unique_categories) // actual_k
        k_folds_cats = [unique_categories[i * fold_size: (i + 1) * fold_size] for i in range(actual_k)]

        # Put any remainder into the last fold
        if len(unique_categories) % actual_k != 0:
            k_folds_cats[-1].extend(unique_categories[actual_k * fold_size:])

        print(f"🔀 {target_layer} has {len(unique_categories)} categories, split into {actual_k} folds for cross‑validation.")

        # 3. Assemble the config structure for the current layer
        config = {
            "dataset_info": f"K-Fold OOD configuration for layer {target_layer}",
            "target_layer": target_layer,
            "total_categories_in_layer": len(unique_categories),
            "total_folds": actual_k
        }

        for i in range(actual_k):
            round_idx = i + 1

            # OOD categories for the current round (completely held out)
            adv_cats = k_folds_cats[i]

            # Remaining categories are used as known classes (Train)
            train_cats = []
            for j in range(actual_k):
                if i != j:
                    train_cats.extend(k_folds_cats[j])

            # 4. Based on the split, filter out the specific species dictionaries belonging to each set
            train_dict = {k: v for k, v in global_taxonomy.items() if v[target_layer] in train_cats}
            adv_dict = {k: v for k, v in global_taxonomy.items() if v[target_layer] in adv_cats}

            config[f"round_{round_idx}"] = {
                "description": f"Adversarial test set consists of the {len(adv_cats)} held‑out {target_layer} categories; the rest are training.",
                "held_out_categories": adv_cats,  # Record the specific names held out in this fold
                "train_metrics": {
                    "total_species_included": len(train_dict),
                    "taxonomy_summary": extract_taxonomy_summary(train_dict)
                },
                "train_splits": extract_branching_splits(train_dict),

                "adversarial_metrics": {
                    "total_species_included": len(adv_dict),
                    "taxonomy_summary": extract_taxonomy_summary(adv_dict)
                },
                "adversarial_splits": extract_branching_splits(adv_dict)
            }

        # 5. Save the independent config file for the current layer
        output_file = os.path.join(OUTPUT_DIR, f"auto_kfold_experiment_config_{target_layer}.json")
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        print(f"🎉 Success! Config file for {target_layer} saved to: {output_file}")


if __name__ == "__main__":
    main()