"""Generic framework state is passed through; cache only a proven explicit workflow."""

from longcache.integrations.langchain import CachedLLM


class ExampleModel:
    def invoke(self, query, **kwargs):
        return "complete answer"


def main():
    # v0.2 does not infer hidden model settings, memory, or a replay codec.
    wrapper = CachedLLM(ExampleModel())
    print(wrapper.invoke("return policy"))
    print(
        "Generic LangChain wrappers pass through in v0.2; see safe_reuse.py for explicit caching."
    )


if __name__ == "__main__":
    main()
