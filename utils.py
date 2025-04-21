import zipfile, torch
from dsa_helpers import imread
import numpy as np
import cv2 as cv
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
import pandas as pd
from geopandas import GeoDataFrame
from datasets.utils.logging import disable_progress_bar

from shapely.affinity import translate
from shapely.geometry import Polygon

from dsa_helpers.ml.datasets.utils import create_segformer_segmentation_dataset
from dsa_helpers.ml.transforms.segformer_transforms import val_transforms
from dsa_helpers.tile_utils import mask_to_shapely


def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def mask_to_geojson(
    mask: str | np.ndarray,
    x_offset: int = 0,
    y_offset: int = 0,
    exclude_labels: int | list[int] = 0,
    min_area: int = 0,
) -> list[Polygon, int]:
    """
    Extract contours from a label mask and convert them into shapely
    polygons.

    Args:
        mask (str | np.ndarray): Path to the mask image or the mask.
        x_offset (int): Offset to add to x coordinates of polygons.
        y_offset (int): Offset to add to y coordinates of polygons.
        exclude_labels (int | list[int]): Label(s) to exclude from the
            output.

    Returns:
        list[Polygon, int]: List of polygons and their corresponding
            labels.

    """
    if isinstance(exclude_labels, int):
        exclude_labels = [exclude_labels]

    if isinstance(mask, str):
        mask = imread(mask, grayscale=True)

    # Find unique labels (excluding background 0)
    labels = [label for label in np.unique(mask) if label not in exclude_labels]

    polygons = []  # Track all polygons.

    # Loop through unique label index.
    for label in labels:
        # Filter to mask for this label.
        label_mask = (mask == label).astype(np.uint8)

        # Find contours.
        contours, hierarchy = cv.findContours(
            label_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        # Process the contours.
        polygons_dict = {}

        for idx, (contour, h) in enumerate(zip(contours, hierarchy[0])):
            if len(contour) > 3:
                if idx not in polygons_dict:
                    polygons_dict[idx] = {"holes": []}

                if h[3] == -1:
                    polygons_dict[idx]["polygon"] = contour.reshape(-1, 2)
                else:
                    polygons_dict[h[3]]["holes"].append(contour.reshape(-1, 2))

        # Now that we know the polygon and the holes, create a Polygon object for each.
        for data in polygons_dict.values():
            if "polygon" in data:
                polygon = Polygon(data["polygon"], holes=data["holes"])

                # Shift the polygon by the offset.
                polygon = translate(polygon, xoff=x_offset, yoff=y_offset)

                # Skip small polygons.
                if polygon.area >= min_area:
                    polygons.append([polygon, int(label)])

    return polygons


def merge_tile_masks(
    tile_list: list, background_label: int | None = 0
) -> GeoDataFrame:
    """
    Merge the tile masks into a single mask for a large image.

    Args:
        tile_list (list): List of three-lenght tuples containing:
            (1) fp or array for mask, (2) x-coordinate for tile, (3)
            y-coordinate for tile.
        background_label (int | None): Label value of the background class,
            which is ignored. Default is 0. If None then all labels are
            considered.

    Returns:
        GeoDataFrame: GeoDataFrame with the merged mask.

    """
    polygons_and_labels = []

    # Process each tile.
    for tile_info in tile_list:
        tile, x, y = tile_info

        # Process the mask by converting it to polygons.
        polygons_and_labels.extend(
            mask_to_shapely(
                tile, x_offset=x, y_offset=y, background_label=background_label
            )
        )

    # Convert polygons and labels into a GeoDataFrame.
    gdf = GeoDataFrame(polygons_and_labels, columns=["geometry", "label"])

    # Ensure geometries are valid (optional but recommended)
    gdf["geometry"] = gdf["geometry"].buffer(0)

    return gdf


def evaluate_model(
    model,
    df,
    id2label,
    tile_size: int,
    group_col: str = "item_id",
    batch_size: int = 8,
    background_label: str = "Background",
):
    """Evaulate a SegFormer for semantic segmentation model on a
    dataset, by stiching tiles back together to their original image.

    Args:
        model (SegformerForSemanticSegmentation): SegFormer model.
        df (DataFrame): Tile dataframe.
        id2label (dict): Mapping from class id to label.
        tile_size (int): Size of the tiles.
        group_col (str): Column name to group the tiles by.
            Default is "item_id".
        batch_size (int): Batch size for evaluation. Default is 8.
        background_label (str): Label for the background class, which is
            ignored. Default is "Background".

    Returns:
        dict: Dictionary of IoUs for each class and the average IoU.

    """
    disable_progress_bar()

    label2id = {label: idx for idx, label in id2label.items()}

    # Create trainer.
    training_args = TrainingArguments(
        ".",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        disable_tqdm=True,
    )

    # Create trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
    )

    groups = list(df[group_col].unique())

    # Track IoUs for each class.
    ious = {label: [] for label in label2id if label != background_label}

    for group in tqdm(groups, desc="Evaluating groups"):
        # Get the tiles in this group.
        group_df = df[df[group_col] == group]

        pred_gdfs = []
        true_gdfs = []

        # Loop through the dataframe in batches.
        for i in range(0, len(group_df), batch_size):
            # Get the batch.
            batch_df = group_df.iloc[i : i + batch_size]
            x_list = batch_df["x"].tolist()
            y_list = batch_df["y"].tolist()

            # Great dataset from the batch.
            dataset = create_segformer_segmentation_dataset(
                batch_df, transforms=val_transforms
            )

            # Predict on the dataset.
            with torch.no_grad():
                out = trainer.predict(dataset)

                logits = out[0]  # (batch_size, num_classes, h, w)
                true = out[1]  # (batch_size, h, w)

            # Take argmax to get the predictions.
            preds = logits.argmax(axis=1)

            # Resize to tile size.
            preds = np.array(
                [
                    cv.resize(
                        pred,
                        (tile_size, tile_size),
                        interpolation=cv.INTER_NEAREST,
                    )
                    for pred in preds
                ]
            )

            # Create a tile list for batch.
            pred_tile_list = [
                (pred, x, y) for x, y, pred in zip(x_list, y_list, preds)
            ]
            true_tile_list = [
                (mask, x, y) for x, y, mask in zip(x_list, y_list, true)
            ]

            # Extract contours from the masks.
            pred_gdfs.append(
                merge_tile_masks(pred_tile_list, background_label=0)
            )
            true_gdfs.append(
                merge_tile_masks(true_tile_list, background_label=0)
            )

        # Concatenate the GeoDataFrames.
        pred_gdf = pd.concat(pred_gdfs)
        true_gdf = pd.concat(true_gdfs)

        # Dissolve by label to merge geometries
        pred_gdf = pred_gdf.dissolve(by="label").reset_index()
        true_gdf = true_gdf.dissolve(by="label").reset_index()

        for idx, label in id2label.items():
            if label == "Background":
                continue
            pred_label_gdf = pred_gdf[pred_gdf["label"] == idx]
            true_label_gdf = true_gdf[true_gdf["label"] == idx]

            # If either the prediction or the ground truth is empty, the IoU is 0.
            if pred_label_gdf.empty or true_label_gdf.empty:
                ious[label].append(0)
            else:
                # Calculate the IoU.
                geom1 = pred_label_gdf.iloc[0]["geometry"]
                geom2 = true_label_gdf.iloc[0]["geometry"]

                intersection_area = geom1.intersection(geom2).area
                union = geom1.union(geom2).area

                ious[label].append(intersection_area / union)

    # Convert the IoUs to a dataframe.
    ious_df = pd.DataFrame(ious)

    # Add an average column which is the average of all other columns.
    ious_df["Average"] = ious_df.mean(axis=1)

    ious_df.loc["Average"] = ious_df.mean()

    # Take the last row and convert to dictionary with labels being the column values.
    ious_dict = ious_df.loc["Average"].to_dict()

    return ious_dict
