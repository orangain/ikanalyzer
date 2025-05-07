import base64
import hashlib
import json
import logging
import os

import cv2
from openai import OpenAI
from openai.types.responses import ParsedResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

cache_dir = os.path.join("cache", "openai")


name_prompt = """
8枚の入力画像に書かれた名前をnames属性に8要素の配列として返してください。名前は記号だけで構成されていることもありますが、書かれたそのままを返してください。
"""


class ExtractPlayerNamesResult(BaseModel):
    names: list[str]


def extract_player_names(
    api_key: str | None,
    player_name_images: list[cv2.typing.MatLike],
    intermediate_path: str,
) -> ExtractPlayerNamesResult | None:
    image_inputs = [
        {"type": "input_image", "image_url": image_url_of(image)}
        for image in player_name_images
    ]
    input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": name_prompt},
            ]
            + image_inputs,
        },
    ]

    result = openai_parse_with_cache(
        api_key=api_key,
        model="gpt-4o-mini",
        text_format=ExtractPlayerNamesResult,
        input=input,
        intermediate_path=intermediate_path,
    )
    return result


class ExtractPlateNamesResult(BaseModel):
    names: list[str]


def extract_plate_names(
    api_key: str | None,
    enemy_plate_images: list[cv2.typing.MatLike],
    enemy_player_names: list[str],
    intermediate_path: str,
) -> ExtractPlateNamesResult | None:
    plate_prompt = f"""
4枚の入力画像に書かれた名前をnames属性に4要素の配列として返してください。名前の候補は以下の4つです。候補以外の名前を返すのは禁止されています。

```
{enemy_player_names[0]}
{enemy_player_names[1]}
{enemy_player_names[2]}
{enemy_player_names[3]}
```
"""

    image_inputs = [
        {"type": "input_image", "image_url": image_url_of(image)}
        for image in enemy_plate_images
    ]
    input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": plate_prompt},
            ]
            + image_inputs,
        },
    ]

    result = openai_parse_with_cache(
        api_key=api_key,
        model="gpt-4o-mini",
        text_format=ExtractPlateNamesResult,
        input=input,
        intermediate_path=intermediate_path,
    )
    return result


def openai_parse_with_cache[TextFormatT: BaseModel](
    api_key: str | None,
    model: str,
    text_format: TextFormatT,
    input: list,
    intermediate_path: str,
) -> ParsedResponse[TextFormatT] | None:
    cache_path = cache_path_of(intermediate_path, input)
    cached_result = read_from_cache(cache_path)
    if cached_result is not None:
        return text_format.model_validate(cached_result)

    if api_key is None:
        return None
    client = OpenAI(api_key=api_key)
    response = client.responses.parse(
        model=model,
        text_format=text_format,
        input=input,
    )

    logger.info(
        f"Response: id={response.id}, error={response.error}, input_tokens={response.usage.input_tokens}, output_tokens={response.usage.output_tokens}"
    )
    result = response.output_parsed

    write_to_cache(cache_path, result.model_dump())
    return result


def image_url_of(image: cv2.typing.MatLike) -> str:
    ret, encoded = cv2.imencode(".png", image)
    if not ret:
        raise ValueError("Failed to encode image")
    # Convert the image to a URL format
    image_url = f"data:image/png;base64,{base64.b64encode(encoded).decode("utf-8")}"
    return image_url


def cache_path_of(intermediate_path: str, input: list) -> str:
    cache_key = hashlib.sha256(json.dumps(input).encode("utf-8")).hexdigest()
    return os.path.join(
        intermediate_path,
        "openai_cache",
        f"{cache_key}.json",
    )


def read_from_cache(cache_path: str) -> dict | None:
    if os.path.exists(cache_path):
        logger.info(f"Cache hit: {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    logger.info(f"Cache miss: {cache_path}")
    return None


def write_to_cache(cache_path: str, data: dict):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
