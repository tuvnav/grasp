from openai.lib.streaming.chat import ChatCompletionStreamState

from grasp.configs import ModelConfig
from grasp.model.openai import OpenAICompletionsModel


class _StreamingCreate:
    # requests the chat completion as a stream and assembles the chunks into
    # the same ChatCompletion object a non-streaming request returns
    def __init__(self, create) -> None:
        self._create = create

    def __call__(self, **kwargs):
        stream = self._create(
            **kwargs,
            stream=True,
            stream_options={"include_usage": True},
        )
        state = ChatCompletionStreamState()
        for chunk in stream:
            state.handle_chunk(chunk)
        return state.get_final_completion()


class OpenAICompletionsStreamModel(OpenAICompletionsModel):
    """
    Chat completions model that streams its responses. Some OpenAI-compatible
    gateways (e.g. OpenWebUI with gpt-6-astra) drop the tool calls of a
    non-streaming response if the model writes text before them; in the
    streamed response they arrive.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        completions = self.client.chat.completions
        completions.create = _StreamingCreate(completions.create)  # type: ignore
