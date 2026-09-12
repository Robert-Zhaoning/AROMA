import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image


MANIFEST_PATH = Path(
    "data/tallyqa_natural_ood_v1/manifest.jsonl"
)

IMAGE_ROOT = Path(
    "data/tallyqa_natural_ood_v1/images"
)

EXPECTED_MANIFEST_SHA256 = (
    "068a489a02d499da6eb0b837a81c8817583b64b51c23f944c77645420fe2c32c"
)

VG_BASE_URL = (
    "https://cs.stanford.edu/people/rak248/"
)

USER_AGENT = (
    "Mozilla/5.0 AROMA-research-image-fetch"
)


def sha256_file(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_manifest():

    observed_hash = (
        sha256_file(
            MANIFEST_PATH
        )
    )

    if (
        observed_hash
        !=
        EXPECTED_MANIFEST_SHA256
    ):

        raise RuntimeError(
            "Frozen manifest SHA256 mismatch.\n"
            f"Expected: {EXPECTED_MANIFEST_SHA256}\n"
            f"Observed: {observed_hash}"
        )

    rows = [
        json.loads(line)
        for line in MANIFEST_PATH
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]

    if len(rows) != 4000:

        raise RuntimeError(
            f"Expected 4000 manifest rows, "
            f"found {len(rows)}."
        )

    paths = [
        str(
            row["image"]
        )
        for row in rows
    ]

    if len(set(paths)) != 4000:

        raise RuntimeError(
            "Manifest image paths are "
            "not globally unique."
        )

    for path in paths:

        if not (
            path.startswith(
                "VG_100K/"
            )
            or
            path.startswith(
                "VG_100K_2/"
            )
        ):

            raise RuntimeError(
                "Unexpected Visual Genome "
                f"path: {path}"
            )

        if ".." in path:

            raise RuntimeError(
                "Unsafe image path."
            )

    return rows


def verify_image(path):

    try:

        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:

            width, height = (
                image.size
            )

            if (
                width <= 0
                or
                height <= 0
            ):
                return False

        return True

    except Exception:

        return False


def build_url(relative_path):

    return (
        VG_BASE_URL
        +
        relative_path
    )


def download_one(relative_path):

    target = (
        IMAGE_ROOT
        /
        relative_path
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        target.exists()
        and
        verify_image(
            target
        )
    ):

        return {
            "image":
                relative_path,

            "status":
                "existing",

            "bytes":
                target.stat().st_size,
        }

    if target.exists():
        target.unlink()

    url = build_url(
        relative_path
    )

    last_error = None

    for attempt in range(5):

        tmp = target.with_suffix(
            target.suffix
            +
            ".part"
        )

        try:

            headers = {
                "User-Agent":
                    USER_AGENT,
            }

            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=90,
                allow_redirects=True,
            ) as response:

                response.raise_for_status()

                content_type = (
                    response.headers
                    .get(
                        "Content-Type",
                        ""
                    )
                    .lower()
                )

                if (
                    content_type
                    and
                    "image" not in content_type
                    and
                    "octet-stream"
                    not in content_type
                ):

                    raise RuntimeError(
                        "Unexpected Content-Type: "
                        f"{content_type}"
                    )

                with tmp.open(
                    "wb"
                ) as fout:

                    for block in (
                        response.iter_content(
                            chunk_size=
                                1024 * 1024
                        )
                    ):

                        if block:

                            fout.write(
                                block
                            )

            if not verify_image(
                tmp
            ):

                raise RuntimeError(
                    "Downloaded file failed "
                    "image verification."
                )

            tmp.replace(
                target
            )

            return {
                "image":
                    relative_path,

                "url":
                    url,

                "status":
                    "downloaded",

                "bytes":
                    target.stat().st_size,
            }

        except Exception as exc:

            last_error = exc

            if tmp.exists():
                tmp.unlink()

            time.sleep(
                min(
                    2 ** attempt,
                    16,
                )
            )

    return {
        "image":
            relative_path,

        "url":
            url,

        "status":
            "failed",

        "error":
            repr(
                last_error
            ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    print("=" * 104)

    print(
        "TALLYQA FROZEN VISUAL-GENOME "
        "DIRECT IMAGE FETCHER"
    )

    print("=" * 104)

    rows = load_manifest()

    requested = [
        str(
            row["image"]
        )
        for row in rows
    ]

    print(
        "Frozen manifest SHA256:",
        EXPECTED_MANIFEST_SHA256,
    )

    print(
        "Frozen manifest images:",
        len(requested),
    )

    print(
        "VG source root:",
        VG_BASE_URL,
    )

    if args.limit is not None:

        requested = (
            requested[
                :args.limit
            ]
        )

    print(
        "Requested this run:",
        len(requested),
    )

    print(
        "Workers:",
        args.workers,
    )

    IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    with ThreadPoolExecutor(
        max_workers=
            args.workers
    ) as executor:

        futures = {
            executor.submit(
                download_one,
                relative,
            ):
                relative

            for relative
            in requested
        }

        completed = 0

        for future in (
            as_completed(
                futures
            )
        ):

            result = (
                future.result()
            )

            results.append(
                result
            )

            completed += 1

            if (
                completed % 100 == 0
                or
                completed
                ==
                len(requested)
            ):

                failed = sum(
                    r["status"]
                    ==
                    "failed"

                    for r in results
                )

                print(
                    f"{completed}/"
                    f"{len(requested)} "
                    f"completed | "
                    f"failed={failed}"
                )

    downloaded = sum(
        r["status"]
        ==
        "downloaded"

        for r in results
    )

    existing = sum(
        r["status"]
        ==
        "existing"

        for r in results
    )

    failures = [
        r
        for r in results
        if r["status"]
        ==
        "failed"
    ]

    print(
        "\n" + "=" * 104
    )

    print(
        "IMAGE FETCH SUMMARY"
    )

    print("=" * 104)

    print(
        "Downloaded:",
        downloaded,
    )

    print(
        "Existing  :",
        existing,
    )

    print(
        "Failed    :",
        len(
            failures
        ),
    )

    if failures:

        print(
            "\nFailure examples:"
        )

        for failure in (
            failures[:20]
        ):

            print(
                json.dumps(
                    failure,
                    ensure_ascii=False,
                )
            )

        raise RuntimeError(
            f"{len(failures)} image "
            "downloads failed."
        )

    print(
        "\nTALLYQA IMAGE FETCH: PASS"
    )


if __name__ == "__main__":
    main()
