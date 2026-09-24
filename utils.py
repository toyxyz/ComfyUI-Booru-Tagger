# Original code: https://github.com/pythongosssss/ComfyUI-WD14-Tagger/blob/main/pysssss.py
import os
import json
import asyncio
import re
import aiohttp
from tqdm import tqdm

config = None


def is_logging_enabled():
    return get_extension_config().get("logging", False)


def log(message, type=None, always=False):
    if not always and not is_logging_enabled():
        return

    if type is not None:
        message = f"[{type}] {message}"

    name = get_extension_config()["name"]

    print(f"(Booru Tagger:{name}) {message}")


def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(os.path.abspath(__file__))
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def get_extension_config(reload=False):
    global config
    if not reload and config is not None:
        return config

    config_path = get_ext_dir("models.json")

    if not os.path.exists(config_path):
        log("Missing models.json, this extension may not work correctly. Please reinstall the extension.",
            type="ERROR", always=True)
        print(f"Extension path: {get_ext_dir()}")
        return {"name": "Unknown", "version": -1}
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return config


def init(check_imports):
    log("Init")

    if check_imports is not None:
        import importlib.util
        for imp in check_imports:
            spec = importlib.util.find_spec(imp)
            if spec is None:
                log(f"{imp} is required, please check requirements are installed.", type="ERROR", always=True)
                return False

    return True


async def download_to_file(url, destination, session=None, progress_cb=None):
    """Download atomically, retrying interrupted transfers with HTTP Range.

    Args:
        session: optional aiohttp.ClientSession to reuse, e.g. one that already
            carries the HuggingFace Authorization header. When omitted, a session
            is created and closed inside this function.
        progress_cb: optional ``callable(total, downloaded)``. Both values refer
            to the complete file, including bytes already held in ``.part``.
    """
    close_session = session is None
    if close_session:
        session = aiohttp.ClientSession()

    proxy = (os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
             or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"))
    proxy_auth = None
    if proxy:
        proxy_auth = aiohttp.BasicAuth(os.getenv("PROXY_USER", ""), os.getenv("PROXY_PASS", ""))

    tmp_destination = destination + ".part"
    meta_destination = tmp_destination + ".json"
    try:
        for attempt in range(5):
            offset = os.path.getsize(tmp_destination) if os.path.exists(tmp_destination) else 0
            try:
                with open(meta_destination, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                meta = {}
            if offset and meta.get("url") != url:
                offset = 0  # Unknown or changed source: do not append to it.

            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
                if meta.get("validator"):
                    headers["If-Range"] = meta["validator"]
            response = None
            try:
                async with session.get(url, proxy=proxy, proxy_auth=proxy_auth,
                                       headers=headers) as response:
                    response.raise_for_status()
                    if response.status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)",
                                             response.headers.get("Content-Range", ""))
                        if not offset or not match or int(match.group(1)) != offset:
                            raise ValueError(f"Invalid Content-Range while resuming {url}")
                        total = int(match.group(3)) if match.group(3) != "*" else 0
                    else:
                        # Range was ignored or If-Range detected a changed file.
                        offset = 0
                        total = int(response.headers.get("Content-Length", 0))

                    validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
                    if validator and validator.startswith("W/"):
                        validator = response.headers.get("Last-Modified")
                    with open(meta_destination, "w", encoding="utf-8") as f:
                        json.dump({"url": url, "validator": validator}, f)
                    downloaded = offset
                    if progress_cb is not None:
                        progress_cb(total, downloaded)
                    with tqdm(unit="B", unit_scale=True, miniters=1,
                              desc=url.split("/")[-1], total=total or None,
                              initial=offset) as progressbar:
                        with open(tmp_destination, "ab" if offset else "wb") as f:
                            async for chunk in response.content.iter_chunked(1024 * 1024):
                                f.write(chunk)
                                downloaded += len(chunk)
                                progressbar.update(len(chunk))
                                if progress_cb is not None:
                                    progress_cb(total, downloaded)
                    if total and downloaded != total:
                        raise aiohttp.ClientPayloadError(
                            f"Incomplete download: {downloaded} of {total} bytes")
                    os.replace(tmp_destination, destination)
                    os.remove(meta_destination)
                    return
            except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError,
                    asyncio.TimeoutError) as err:
                if attempt == 4:
                    raise
                log(f"Download interrupted ({err}); retrying from saved bytes", "WARN", True)
            except aiohttp.ClientResponseError as err:
                if err.status == 416 and offset and response is not None:
                    complete = re.fullmatch(r"bytes \*/(\d+)",
                                            response.headers.get("Content-Range", ""))
                    if complete and int(complete.group(1)) == offset:
                        os.replace(tmp_destination, destination)
                        os.remove(meta_destination)
                        return
                    # The remote file changed or the partial file is unusable.
                    os.remove(tmp_destination)
                    os.remove(meta_destination)
                    if attempt < 4:
                        continue
                if err.status not in (408, 429, 500, 502, 503, 504) or attempt == 4:
                    raise
                log(f"Download returned HTTP {err.status}; retrying", "WARN", True)
            await asyncio.sleep(min(2 ** attempt, 8))
    finally:
        if close_session:
            await session.close()
