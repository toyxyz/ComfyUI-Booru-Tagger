from .utils import init
from .nodes import BooruTaggerExtension

WEB_DIRECTORY = "./web"
_route_registered = False


def _register_model_info_route():
    global _route_registered
    if _route_registered:
        return

    from aiohttp import web
    from server import PromptServer
    from .model_info import get_model_info

    @PromptServer.instance.routes.get("/booru-tagger/model-info")
    async def model_info(request):
        try:
            info = get_model_info(request.query.get("model_name", ""))
        except KeyError:
            return web.json_response({"error": "Unknown model"}, status=404)
        return web.json_response(info)

    _route_registered = True

async def comfy_entrypoint() -> BooruTaggerExtension:
    if init(check_imports=["onnxruntime"]):
        _register_model_info_route()
        return BooruTaggerExtension()
    else:
        raise ImportError("onnxruntime is required.")

