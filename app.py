import spaces
import json
import subprocess
from llama_cpp import Llama
from llama_cpp_agent import LlamaCppAgent, MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppPythonProvider
from llama_cpp_agent.chat_history import BasicChatHistory
from llama_cpp_agent.chat_history.messages import Roles
import gradio as gr
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="OEvortex/HelpingAI2-6B",
    filename="helpingai-6b-q4_k_m.gguf",
    local_dir="./models"
)
hf_hub_download(
    repo_id="Abhaykoul/HelpingAI2-4x6B",
    filename="helpingai2-4x6b-q4_k_m.gguf",
    local_dir="./models"
)


llm = None
llm_model = None

@spaces.GPU(duration=120)
def respond(
    message,
    history: list[tuple[str, str]],
    model,
    system_message,
    max_tokens,
    temperature,
    top_p,
    top_k,
    repeat_penalty,
):
    chat_template = MessagesFormatterType.LLAMA_3

    global llm
    global llm_model
    
    if llm is None or llm_model != model:
        llm = Llama(
            model_path=f"models/{model}",
            flash_attn=True,
            n_gpu_layers=81,
            n_batch=1024,
            n_ctx=8192,
        )
        llm_model = model

    provider = LlamaCppPythonProvider(llm)

    agent = LlamaCppAgent(
        provider,
        system_prompt=f"{system_message}",
        predefined_messages_formatter_type=chat_template,
        debug_output=True
    )
    
    settings = provider.get_provider_default_settings()
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p
    settings.max_tokens = max_tokens
    settings.repeat_penalty = repeat_penalty
    settings.stream = True

    messages = BasicChatHistory()

    for msn in history:
        user = {
            'role': Roles.user,
            'content': msn[0]
        }
        assistant = {
            'role': Roles.assistant,
            'content': msn[1]
        }
        messages.add_message(user)
        messages.add_message(assistant)
    
    stream = agent.get_chat_response(
        message,
        llm_sampling_settings=settings,
        chat_history=messages,
        returns_streaming_generator=True,
        print_output=False
    )
    
    outputs = ""
    for output in stream:
        outputs += output
        yield outputs

description = "Defualt to 6B in Additional Inputs you can change model"


demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Dropdown([
                'helpingai-6b-q4_k_m.gguf',
                'helpingai2-4x6b-q4_k_m.gguf',
            ],
            value="helpingai-6b-q4_k_m.gguf",
            label="Model"
        ),
        gr.Textbox(value="You are HelpingAI a emotional AI always answer my question in HelpingAI style", label="System message"),
        gr.Slider(minimum=1, maximum=4096, value=2048, step=1, label="Max tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p",
        ),
        gr.Slider(
            minimum=0,
            maximum=100,
            value=40,
            step=1,
            label="Top-k",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.1,
            step=0.1,
            label="Repetition penalty",
        ),
    ],
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
    submit_btn="Send",
    title="Chat with HelpingAI using llama.cpp", 
    description=description,
    chatbot=gr.Chatbot(
        scale=1, 
        likeable=False,
        show_copy_button=True
    )
)

if __name__ == "__main__":
    demo.launch()