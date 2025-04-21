"""Train a model with the University of Kentucky dataset (annotated for different tissue types).
The model trained is the Segformer for Semantic Segmentation.
python3 2.2-train-uk-model-20.py --dataset-dir /home/bhavesh/ml-work/tiles --save-dir /home/bhavesh/ml-work/20imgs/models
"""

import warnings, json, shutil, torch
from argparse import ArgumentParser
from pprint import pprint
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import SegformerForSemanticSegmentation, TrainingArguments, Trainer
from utils import evaluate_model
from time import perf_counter

from dsa_helpers.ml.datasets.utils import create_segformer_segmentation_dataset
from dsa_helpers.ml.transforms.segformer_transforms import train_transforms, val_transforms
from dsa_helpers.ml.metrics import mean_iou
from dsa_helpers.ml.callbacks import MetricsLoggerCallback
from dsa_helpers.girder_utils import login


warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector",
    category=UserWarning,
)


def merge_tile_metadata():
    tile_metadata_1 = Path("/home/bhavesh/ml-work/tiles/tile_metadata.csv")
    tile_metadata_2 = Path("/home/bhavesh/ml-work/20imgs/tiles1/tile_metadata1.csv")

    df1 = pd.read_csv(tile_metadata_1)
    df2 = pd.read_csv(tile_metadata_2)

    merged_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates()

    merged_output_path = Path("/home/bhavesh/ml-work/20imgs/tiles_merged/tile_metadata.csv")
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(merged_output_path, index=False)

    return merged_output_path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--dsa-api-url", default="http://bdsa.pathology.emory.edu:8080/api/v1")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-accumulation-steps", type=int, default=100)
    parser.add_argument("--max-val-size", type=int, default=500)
    return parser.parse_args()


def get_folder_by_path(gc, path_parts):
    parent_type = "collection"
    parent_id = None
    current_folder = None

    for i, part in enumerate(path_parts):
        if i == 0:
            for collection in gc.listCollection():
                if collection["name"] == part:
                    parent_id = collection["_id"]
                    parent_type = "collection"
                    break
        else:
            folders = gc.listFolder(parent_id, parentFolderType=parent_type)
            for f in folders:
                if f["name"] == part:
                    parent_id = f["_id"]
                    parent_type = "folder"
                    current_folder = f
                    break
            else:
                current_folder = gc.createFolder(parent_id, part, parentType=parent_type)
                parent_id = current_folder["_id"]
                parent_type = "folder"
    return current_folder


def main(args):
    pprint(vars(args))
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir)
    dataset_dir = Path(args.dataset_dir)

    gc = login(args.dsa_api_url)

    merged_tile_metadata_path = merge_tile_metadata()
    tile_df = pd.read_csv(merged_tile_metadata_path)
    print(f"Merged tile metadata CSV saved at: {merged_tile_metadata_path}")

    wsi_item_ids = sorted(list(tile_df["item_id"].unique()))

    train_ids, val_ids = train_test_split(
        wsi_item_ids, test_size=args.val_frac, random_state=args.random_state
    )

    train_tiles_df = tile_df[tile_df["item_id"].isin(train_ids)]
    val_tiles_df = tile_df[tile_df["item_id"].isin(val_ids)]

    print(f"Number of training tiles: {len(train_tiles_df)}")
    print(f"Number of validation tiles: {len(val_tiles_df)}")

    save_dir.mkdir(parents=True, exist_ok=True)
    train_tiles_df.to_csv(save_dir / "train_tiles.csv", index=False)
    val_tiles_df.to_csv(save_dir / "val_tiles.csv", index=False)

    with open(dataset_dir / "args.json", "r") as f:
        tile_args = json.load(f)

    tile_args.update({
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "max_val_size": args.max_val_size,
        "dataset_dir": str(dataset_dir)
    })

    val_tiles_df = val_tiles_df.sample(frac=1, random_state=args.random_state)[: args.max_val_size]

    train_dataset = create_segformer_segmentation_dataset(train_tiles_df, transforms=train_transforms)
    val_dataset = create_segformer_segmentation_dataset(val_tiles_df, transforms=val_transforms)

    label2id = tile_args["label2idx"]
    id2label = {int(v): k for k, v in label2id.items()}

    print("Label to ID mapping:")
    pprint(label2id)

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", id2label=id2label, label2id=label2id
    ).to(device)

    training_args = TrainingArguments(
        str(save_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=mean_iou(label2id),
        callbacks=[MetricsLoggerCallback],
    )

    start_t = perf_counter()
    _ = trainer.train()
    tile_args['training_time'] = perf_counter() - start_t
    tile_args["train_item_ids"] = train_ids
    tile_args["val_item_ids"] = val_ids

    model_name = "segformer-uk-model-17-mag10"

    for d in save_dir.iterdir():
        if d.stem.startswith("checkpoint"):
            d.rename(save_dir / model_name)

    val_results = evaluate_model(
        model, val_tiles_df, id2label, tile_args['tile_size'],
        batch_size=args.batch_size
    )
    tile_args['iou_metrics'] = val_results

    shutil.make_archive(save_dir / model_name, "zip", save_dir / model_name)

    model_target_path = ["Emory ADRC Cohorts", "Bhavesh", "model"]
    target_folder = get_folder_by_path(gc, model_target_path)

    model_item = gc.createItem(target_folder["_id"], save_dir.stem, metadata=tile_args)
    gc.uploadFileToItem(model_item["_id"], f"{model_name} / checkpoint.zip")


if __name__ == "__main__":
    main(parse_args())
