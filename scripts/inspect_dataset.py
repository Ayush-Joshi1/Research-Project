"""
Inspect BEIR NQ queries and qrels (inspection only).

This script is a read-only inspection step to discover the dataset configuration,
number of queries, available splits, and qrels structure. It does NOT download
the full corpus, create embeddings, or build an index.
"""

from pprint import pprint
from datasets import load_dataset, load_dataset_builder


def try_load_builder(dataset_ids):
    # Try loading builder directly; if dataset requires a config name, try common configs.
    config_candidates = [None, "queries", "corpus"]
    for ds_id in dataset_ids:
        for cfg in config_candidates:
            try:
                if cfg is None:
                    print(f"Trying dataset id: {ds_id} (no config)")
                    builder = load_dataset_builder(ds_id)
                else:
                    print(f"Trying dataset id: {ds_id} with config: {cfg}")
                    builder = load_dataset_builder(ds_id, name=cfg)
                print(f"Loaded builder for: {ds_id} (config={cfg})")
                return ds_id, cfg, builder
            except Exception as e:
                print(f"Could not load builder for {ds_id} (config={cfg}): {e}")
                continue
    return None, None, None


def main():
    # Candidate dataset identifiers for SciFact
    candidates = ["BeIR/scifact", "beir/scifact", "scifact/beir", "scifact"]

    ds_id, cfg, builder = try_load_builder(candidates)
    if builder is None:
        print("Failed to locate a compatible BEIR scifact dataset builder on Hugging Face.")
        print("Please verify the correct dataset id (e.g., 'BeIR/scifact') and try again.")
        return

    # Print available configs (if any)
    try:
        configs = list(builder.builder_configs.keys())
    except Exception:
        configs = None
    print("Available configs:", configs)

    # Print splits and sizes
    splits = getattr(builder.info, "splits", None)
    if splits:
        print("Available splits:")
        for name, splitinfo in splits.items():
            print(f" - {name}: {getattr(splitinfo, 'num_examples', 'unknown')} examples")
    else:
        print("No split information available from builder.info")

    # Inspect both 'corpus' and 'queries' splits if available
    for split_name in ["corpus", "queries"]:
        print(f"\n--- Inspecting split: {split_name} ---")
        if splits and split_name in splits:
            num_examples = getattr(splits[split_name], "num_examples", None)
            print(f"Number of examples for split '{split_name}': {num_examples}")
        else:
            print(f"Split '{split_name}' not present in builder metadata.")

        # Try to stream a few examples from the split
        try:
            # If inspecting the corpus split, explicitly request the 'corpus' config
            if split_name == "corpus":
                ds_split = load_dataset(ds_id, "corpus", split="corpus", streaming=True)
            else:
                if cfg:
                    ds_split = load_dataset(ds_id, cfg, split=split_name, streaming=True)
                else:
                    ds_split = load_dataset(ds_id, split=split_name, streaming=True)
            print(f"Features / columns for split '{split_name}':")
            try:
                # Try to get features from builder.info if available
                features = builder.info.features
                pprint(list(features.keys()))
            except Exception:
                print("Could not read features from builder.info for this split.")

            print(f"Example documents from split '{split_name}' (up to 3):")
            examples = []
            for i, item in enumerate(ds_split):
                if i >= 3:
                    break
                examples.append(item)
                print(f"Example {i+1}:")
                pprint(item)
        except Exception as e:
            print(f"Could not stream split '{split_name}': {e}")
            print("Skipping streaming examples for this split.")

    print("\nAttempting to locate qrels / relevance judgments for SciFact...")
    qrels = None
    # Try known qrels dataset ids and common patterns
    qrel_ids = ["BeIR/scifact-qrels", "beir/scifact-qrels", "scifact-qrels", f"{ds_id}-qrels"]
    for qid in qrel_ids:
        try:
            print(f"Trying qrels dataset id: {qid}")
            qrels = load_dataset(qid, streaming=True)
            print(f"Loaded qrels dataset: {qid} (streaming)")
            qrel_ds_id = qid
            break
        except Exception as e:
            print(f"Could not load qrels dataset {qid}: {e}")
            qrels = None
            continue

    # If qrels not loaded, check builder.info or dataset metadata for hints
    if qrels is None:
        print("No separate qrels dataset found with common names. They may be stored as files in the dataset repo.")
    else:
        # qrels may be a DatasetDict-like (mapping of splits) or an iterable dataset
        if hasattr(qrels, "keys"):
            # List available qrel splits
            qrel_splits = list(qrels.keys())
            print("Qrels available splits:", qrel_splits)
            total_qrel_count = 0
            qrels_examples = []
            for split in qrel_splits:
                print(f"\nInspecting qrels split: {split}")
                try:
                    ds_q = load_dataset(qrel_ds_id, split=split, streaming=True)
                    count = 0
                    for i, item in enumerate(ds_q):
                        if i < 3:
                            pprint(item)
                            qrels_examples.append(item)
                        count += 1
                    print(f"Number of records iterated for split '{split}': {count}")
                    total_qrel_count += count
                except Exception as e:
                    print(f"Could not stream qrels split '{split}': {e}")
            print(f"Total qrel records iterated across splits (streaming count): {total_qrel_count}")
        else:
            # iterable dataset returned; iterate to get examples and count
            qrels_examples = []
            count = 0
            try:
                for i, item in enumerate(qrels):
                    if i < 3:
                        pprint(item)
                        qrels_examples.append(item)
                    count += 1
                print(f"Total qrel records iterated (streaming count): {count}")
            except Exception as e:
                print(f"Error iterating qrels in streaming mode: {e}")

        # Try to infer qrel schema from examples
        if qrels_examples:
            keys = set()
            for ex in qrels_examples:
                if isinstance(ex, dict):
                    keys.update(ex.keys())
            if keys:
                print("Qrels fields:", list(keys))
                mapping_keys = [k for k in keys if any(x in k.lower() for x in ["query", "qid"]) ]
                doc_keys = [k for k in keys if any(x in k.lower() for x in ["doc", "pid", "id"]) ]
                print("Possible query-id keys:", mapping_keys)
                print("Possible doc-id keys:", doc_keys)
            else:
                print("Could not infer qrel field names from examples.")

    # If qrels are not available, and examples exist, attempt to show relevant doc ids
    if qrels is None and examples:
        # Some dataset items may include a field with 'relevant' or 'positive' ids
        sample = examples[0]
        possible_keys = [k for k in sample.keys() if "id" in k or "doc" in k or "relevant" in k or "qid" in k]
        print("Possible keys in example for locating ids:", possible_keys)
        # Try common names
        for k in ["query_id", "qid", "id", "query-id"]:
            if k in sample:
                print(f"Found query id under key '{k}': {sample[k]}")
                break


if __name__ == "__main__":
    main()
