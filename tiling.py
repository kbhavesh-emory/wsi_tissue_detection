import large_image
import cv2 as cv
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from dsa_helpers import imwrite, imread
import histomicstk as htk

stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

# specify stains of input image
stains = [
    "hematoxylin",  # nuclei stain
    "eosin",  # cytoplasm stain
    "null",
]  # set to null if input contains only two stains

# create stain matrix
W = np.array([stain_color_map[st] for st in stains]).T


def _proccess_tile_with_masks_from_dsa_annotations(
    ts,
    prepend_name,
    x,
    y,
    scan_tile_size,
    tile_size,
    mag,
    contours,
    sf,
    img_dir,
    mask_dir,
    background_idx,
    edge_thr,
    deconvolve,
):
    # Calculate the x and y in tile magnification.
    x_mag = int(x * sf)
    y_mag = int(y * sf)

    fn = f"{prepend_name}x{x_mag}y{y_mag}.png"

    # Get the tile.
    tile_img = ts.getRegion(
        region={
            "left": x,
            "top": y,
            "right": x + scan_tile_size,
            "bottom": y + scan_tile_size,
        },
        format=large_image.constants.TILE_FORMAT_NUMPY,
        scale={"magnification": mag},
    )[0][:, :, :3]

    if deconvolve:
        # Deconvolve the image.s
        tile_img = htk.preprocessing.color_deconvolution.color_deconvolution(
            tile_img, W
        ).Stains[:, :, 0]

        tile_img = np.stack([tile_img, tile_img, tile_img], axis=-1)

    # Check if the tile area is below threshold.
    h, w = tile_img.shape[:2]

    if h * w / (tile_size * tile_size) < edge_thr:
        return None

    if (h, w) != (tile_size, tile_size):
        # Pad the image with zeroes to make it the desired size.
        tile_img = cv.copyMakeBorder(
            tile_img,
            0,
            tile_size - tile_img.shape[0],
            0,
            tile_size - tile_img.shape[1],
            cv.BORDER_CONSTANT,
            value=background_idx,
        )

    # Create a blank tile mask.
    tile_mask = np.zeros(tile_img.shape[:2], dtype=np.uint8)

    # Loop through contours to draw on tile mask.
    for idx, contour_dict in contours.items():
        # Create a mask for the specific label.
        label_mask = np.zeros(tile_mask.shape, dtype=np.uint8)

        points = contour_dict["points"].copy()
        holes = contour_dict["holes"].copy()

        # For points and hole:
        # (1) Shift by top left of the current tile.
        # (2) Scale by be desired magnification.
        points = [((point - [x, y]) * sf).astype(int) for point in points]

        corrected_holes = []

        for hole in holes:
            if len(hole):
                corrected_holes.append(((hole - [x, y]) * sf).astype(int))

        # Use points to draw contours as positive.
        label_mask = cv.drawContours(label_mask, points, -1, 1, cv.FILLED)

        # Draw in the holes as background class zero.
        label_mask = cv.drawContours(label_mask, holes, -1, 0, cv.FILLED)

        # Apply the label to tile mask where label mask is positive.
        tile_mask[label_mask == 1] = idx

    # Save the image and mask.
    img_fp = img_dir / fn
    imwrite(img_fp, tile_img)
    imwrite(mask_dir / fn, tile_mask, grayscale=True)

    return str(img_fp), x_mag, y_mag


def tile_wsi_with_masks_from_dsa_annotations(
    wsi_fp: str,
    annotation_docs: list[dict],
    label2idx: dict,
    save_dir: str,
    tile_size: int,
    stride: int | None = None,
    mag: float | None = None,
    prepend_name: str = "",
    nproc: int = 1,
    background_idx: int = 0,
    edge_thr: float = 0.25,
    deconvolve: bool = False,
) -> list[str]:
    """Tile a WSI with semantic segmentation label masks created from
    DSA annotations. DSA annotation class labels for elements are
    specified in the "label" dictionary in the "value" key of each
    element.

    Args:
        wsi_fp (str): file path to WSI.
        annotation_docs (list[dict]): list of annotation documents.
        label2idx (dict): mapping of label names to integer indices.
        save_dir (str): directory to save the tiled images.
        tile_size (int): size of the tile images at desired
            magnificaiton.
        stride (int | None, optional): stride of the tiling. If None,
            the stride will be the same as the tile size. Defaults to
            None.
        mag (float | None, optional): magnification level of the WSI.
            If None, the function will use the default magnification
            level. Defaults to None.
        prepend_name (str, optional): prepend name to the created tile
            images and masks. Defaults to "".
        nproc (int, optional): number of processes to use for tiling.
            Defaults to 1.
        background_idx (int, optional): index of the background class.
            Will be used when padding edges of WSI. Defaults to 0.
        edge_thr (float, optional): For tiles at the edge, if most of
            the tile is padded background then it is ignored. If the
            amount of tile in WSI / amount of tile padded is less than
            this threshold, the tile is ignored. Defaults to 0.25.
        deconvolve (bool, optional): whether to deconvolve the tiles
            before saving. Defaults to False.

    Returns:
        list[str]: A list of tuples: (tile file path, x, y coordinates
            of tile at magnification desired).

    """
    # Get the tile source of WSI.
    ts = large_image.getTileSource(str(wsi_fp))

    # Get the metadata of the WSI.
    metadata = ts.getMetadata()

    # Calculate variables based on the desired magnification.
    wsi_mag = metadata["magnification"]

    if stride is None:
        stride = tile_size

    # Calculate the tile size at full resolution (1 if mag is None or the same as scan mag).
    if mag is None:
        scan_tile_size = tile_size
        scan_stride = stride
        mag = wsi_mag
        sf = 1
    else:
        sf = mag / wsi_mag  # scan mag -> desired mag
        scan_tile_size = int(tile_size / sf)
        scan_stride = int(stride / sf)

    # Format annotation documents for drawing on tile masks.
    contours = {idx: {"points": [], "holes": []} for idx in label2idx.values()}

    for ann_doc in annotation_docs:
        # Loop through the elements of the annotation document.
        elements = ann_doc.get("annotation", {}).get("elements", [])

        for element in elements:
            # Check if element has a label.
            label = element.get("label", {}).get("value")

            if label not in label2idx:
                # Skip this element if it is not in the label2idx mapping.
                continue

            idx = label2idx[label]

            element_type = element["type"]

            if element_type == "polyline":
                points = element.get("points", [])

                if len(points):
                    points = np.array(points, dtype=int)[:, :2]
                    contours[idx]["points"].append(points)

                holes = element.get("holes", [])

                if len(holes):
                    for hole in holes:
                        # Scale by the factor.
                        hole = np.array(hole, dtype=int)[:, :2]
                        contours[idx]["holes"].append(hole)

            elif element_type == "rectangle":
                # Convert the rectangle coordinates to polygon ones.
                x1, y1 = element["center"][:2]
                x2 = x1 + element["width"]
                y2 = y1 + element["height"]

                points = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=int
                )

                # Scale by the factor.
                contours[idx]["points"].append(points)

    # Create directory to save tile images and masks.
    save_dir = Path(save_dir)
    img_dir = save_dir / "images"
    mask_dir = save_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    # Get a list of x,y coordinates for the tile top left corners.
    wsi_w, wsi_h = metadata["sizeX"], metadata["sizeY"]

    xys = [
        (x, y)
        for x in range(0, wsi_w, scan_stride)
        for y in range(0, wsi_h, scan_stride)
    ]

    with Pool(nproc) as pool:
        jobs = [
            pool.apply_async(
                _proccess_tile_with_masks_from_dsa_annotations,
                (
                    ts,
                    prepend_name,
                    xy[0],
                    xy[1],
                    scan_tile_size,
                    tile_size,
                    mag,
                    contours,
                    sf,
                    img_dir,
                    mask_dir,
                    background_idx,
                    edge_thr,
                    deconvolve,
                ),
            )
            for xy in xys
        ]

        tile_info = []

        for job in tqdm(jobs, desc="Tiling..."):
            out = job.get()

            if out is not None:
                tile_info.append(job.get())

        return tile_info
