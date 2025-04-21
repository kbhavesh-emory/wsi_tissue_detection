# python3 2.1.2-tile-kentucky-wsis-selected-10.py --save-dir /home/bhavesh/ml-work/10imgs/tiles1 --nproc 25

from argparse import ArgumentParser
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
from config import label2idx
import large_image
from tiling import tile_wsi_with_masks_from_dsa_annotations
from dsa_helpers.girder_utils import login
from glob import glob


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/home/bhavesh/ml-work/10imgs/tiles1",
        help="Directory to save the tiles.",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--mag", type=float, default=10)
    parser.add_argument("--nproc", type=int, default=25)
    parser.add_argument(
        "--dsa-api-url",
        type=str,
        default="http://bdsa.pathology.emory.edu:8080/api/v1",
    )
    parser.add_argument(
        "--annotation-doc-name",
        type=str,
        default="Tissue Compartments Inference",
    )
    parser.add_argument(
        "--dsa-folder-id",
        type=str,
        default="67c5f83165fd0aa5859665b7",
    )
    parser.add_argument(
        "--wsi-dir",
        type=str,
        default="/wsi_archive/APOLLO_NP",
        help="Directory containing WSIs (not glob pattern here)",
    )
    parser.add_argument("--deconvolve", action="store_true")
    return parser.parse_args()


def main(args):
    print("Tiling UK WSIs with label masks from DSA annotations.")
    print("Authenticate girder client:")
    gc = login(args.dsa_api_url)

    # Fetch first 10 annotated items
    annotations = {}
    items = list(gc.listItem(args.dsa_folder_id))[:10]
    for item in tqdm(items, desc="Reading annotations from DSA (max 10):"):
        ann_docs = []
        for ann_doc in gc.get(f"annotation/item/{item['_id']}"):
            if ann_doc.get("annotation", {}).get("name") == args.annotation_doc_name:
                ann_docs.append(ann_doc)
        if ann_docs:
            annotations[item["name"]] = ann_docs

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    wsi_dir = Path(args.wsi_dir)
    nproc = args.nproc

    # Load all WSI files
    print(f"Searching for WSIs under: {wsi_dir}")
    fps = list(wsi_dir.rglob("*"))
    fp_dict = {fp.name: fp for fp in fps if fp.is_file()}

    # Prepare args
    args_dict = vars(args)
    del args_dict["wsi_dir"]
    del args_dict["nproc"]
    del args_dict["save_dir"]
    args_dict["label2idx"] = label2idx.copy()
    if "Background" in label2idx:
        del label2idx["Background"]

    with (save_dir / "args.json").open("w") as fh:
        json.dump(args_dict, fh)

    tile_metadata = []

    for i, (item_name, ann_docs) in enumerate(list(annotations.items())[:10]):
        print(f"({i + 1}/10) Processing {item_name}")
        item_id = ann_docs[0]["itemId"]

        if item_name in fp_dict:
            wsi_fp = fp_dict[item_name]
            print(f"Matched path: {wsi_fp}")
        else:
            print(f"No matching file found for: {item_name}")
            continue

        tile_list = tile_wsi_with_masks_from_dsa_annotations(
            wsi_fp,
            ann_docs,
            label2idx,
            save_dir / wsi_fp.stem,
            args.tile_size,
            mag=args.mag,
            prepend_name=f"{wsi_fp.stem}-",
            nproc=nproc,
            edge_thr=0.0,
            deconvolve=args.deconvolve,
        )

        df = pd.DataFrame(tile_list, columns=["fp", "x", "y"])

        for idx, row in df.iterrows():
            df.at[idx, "mask_fp"] = row["fp"].replace("/images/", "/masks/")

        df["wsi_name"] = [item_name] * len(df)
        df["mag"] = [args.mag] * len(df)
        df["tile_size"] = [args.tile_size] * len(df)
        df["item_id"] = [item_id] * len(df)

        ts = large_image.getTileSource(str(wsi_fp))
        scan_mag = ts.getMetadata()["magnification"]
        df["scan_mag"] = [scan_mag] * len(df)

        tile_metadata.append(df)

    tile_metadata = pd.concat(tile_metadata, ignore_index=True)
    tile_metadata.to_csv(save_dir / "tile_metadata1.csv", index=False)

    print(f"Saved {len(tile_metadata)} tiles.")

if __name__ == "__main__":
    main(parse_args())
