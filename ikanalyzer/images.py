import cv2
import numpy as np


def crop_image(
    image: cv2.typing.MatLike,
    roi: cv2.typing.Rect,
) -> cv2.typing.MatLike:
    x, y, w, h = roi
    return image[y : y + h, x : x + w]


def normalize_image(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    # グレースケールに変換
    image_normalized = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 3値化
    image_normalized = three_level_binarize(image_normalized)
    return image_normalized


def three_level_binarize(gray_img: cv2.typing.MatLike, low_thresh=60, high_thresh=200):
    result = np.full_like(gray_img, 128)  # 中間輝度で初期化

    result[gray_img <= low_thresh] = 0  # 黒
    result[gray_img >= high_thresh] = 255  # 白

    return result


def binarize_image(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    image_binarized = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binarized = cv2.threshold(
        image_binarized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )  # OTSU法による2値化
    return image_binarized


def crop_non_zero_area(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    coords = cv2.findNonZero(image)
    roi = cv2.boundingRect(coords)
    return crop_image(image, roi)


def vstack(images: list[cv2.typing.MatLike], pad_value=0) -> cv2.typing.MatLike:
    max_width = max(img.shape[1] for img in images)

    padded_images = []
    for img in images:
        _, w = img.shape[:2]
        if w < max_width:
            # パディング量を計算
            pad_right = max_width - w

            # 画像に左右パディング追加
            img = cv2.copyMakeBorder(
                img,
                0,
                0,
                0,
                pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=[pad_value] * (img.shape[2] if len(img.shape) == 3 else 1),
            )
        padded_images.append(img)

    # パディング済み画像を縦に結合
    return np.vstack(padded_images)


def hstack(images: list[cv2.typing.MatLike], pad_value=0) -> cv2.typing.MatLike:
    # 最大高さを取得
    max_height = max(img.shape[0] for img in images)

    padded_images = []
    for img in images:
        h, _ = img.shape[:2]
        if h < max_height:
            # パディング量を計算
            pad_bottom = max_height - h

            # 画像に上下パディング追加
            img = cv2.copyMakeBorder(
                img,
                0,
                pad_bottom,
                0,
                0,
                borderType=cv2.BORDER_CONSTANT,
                value=[pad_value] * (img.shape[2] if len(img.shape) == 3 else 1),
            )
        padded_images.append(img)

    # パディング済み画像を横に結合
    return np.hstack(padded_images)


def max_size_of(images: list[cv2.typing.MatLike]) -> tuple[int, int]:
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    return max_height, max_width


def paste_in_center(
    image: cv2.typing.MatLike, size: tuple[int, int], pad_value=0
) -> cv2.typing.MatLike:
    h, w = image.shape[:2]
    oh, ow = size

    assert oh >= h and ow >= w, "Target size must be larger than image size."

    pad_top = (oh - h) // 2
    pad_bottom = oh - h - pad_top
    pad_left = (ow - w) // 2
    pad_right = ow - w - pad_left

    return cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[pad_value] * (image.shape[2] if len(image.shape) == 3 else 1),
    )


def transform_perspective(
    image: cv2.typing.MatLike,
    src_points: tuple[
        cv2.typing.Point, cv2.typing.Point, cv2.typing.Point, cv2.typing.Point
    ],
    size: tuple[int, int],
) -> cv2.typing.MatLike:
    # 1. 台形の4頂点を指定（時計回りまたは反時計回り）
    src_points_matrix = np.float32(src_points)
    # 2. 出力画像のサイズを決める
    width, height = size
    dest_points_matrix = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]
    )
    # 3. 変換行列を求める
    matrix = cv2.getPerspectiveTransform(src_points_matrix, dest_points_matrix)
    # 4. ワープして補正画像を得る
    return cv2.warpPerspective(image, matrix, (width, height))
