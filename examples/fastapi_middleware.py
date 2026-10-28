"""Public, non-personalized fixture route with an explicit v0.2 middleware policy."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from longcache import CacheConfig, LongCache, RequestContext
from longcache.middleware import LongCacheMiddleware

context = RequestContext(
    scope=None,
    operation="public-demo.answer",
    provider="fixture",
    provider_scope="demo-v1",
    model="demo",
    prompt_version="v1",
)
cache = LongCache(CacheConfig(backend="memory", scope_policy="public"))


@asynccontextmanager
async def lifespan(app):
    yield
    cache.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(LongCacheMiddleware, cache=cache, context=context)


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    # A fixed public fixture response; no private/authenticated user data is used.
    return JSONResponse(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "This is the public demo answer."},
                }
            ]
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
