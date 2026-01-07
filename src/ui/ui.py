import os
import shutil
import json
import sys
import time
import gradio as gr
from dataclasses import dataclass
from typing import ClassVar
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from .theme import JS_LIGHT_THEME, CSS
from ..core.prompt.qa_prompt import get_system_prompt
from ..pipeline import LocalRAGPipeline
from ..logger import Logger
#------------------------------------------------------------------------------
@dataclass
class DefaultElement:
    DEFAULT_MESSAGE: ClassVar[dict] = {"text": ""}
    DEFAULT_MODEL: str = ""
    DEFAULT_HISTORY: ClassVar[list] = []
    DEFAULT_DOCUMENT: ClassVar[list] = []

    HELLO_MESSAGE: str = "Hi 👋, how can I help you today?"
    SET_MODEL_MESSAGE: str = "You need to choose LLM model 🤖 first!"
    EMPTY_MESSAGE: str = "You need to enter your message!"
    DEFAULT_STATUS: str = "Ready!"
    CONFIRM_PULL_MODEL_STATUS: str = "Confirm Pull Model!"
    PULL_MODEL_SCUCCESS_STATUS: str = "Pulling model 🤖 completed!"
    PULL_MODEL_FAIL_STATUS: str = "Pulling model 🤖 failed!"
    MODEL_NOT_EXIST_STATUS: str = "Model doesn't exist!"
    PROCESS_DOCUMENT_SUCCESS_STATUS: str = "Processing documents 📄 completed!"
    PROCESS_DOCUMENT_EMPTY_STATUS: str = "Empty documents!"
    ANSWERING_STATUS: str = "Answering!"
    COMPLETED_STATUS: str = "Completed!"
#------------------------------------------------------------------------------
class LLMResponse:
    def __init__(self) -> None:
        pass
    #---
    def _yield_string(self, message: str):
        for i in range(len(message)):
            time.sleep(0.01)
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                [{"role": "assistant", "content": message[: i + 1]}],
                DefaultElement.DEFAULT_STATUS,
            )
    #---
    def welcome(self):
        yield from self._yield_string(DefaultElement.HELLO_MESSAGE)
    #---
    def set_model(self):
        yield from self._yield_string(DefaultElement.SET_MODEL_MESSAGE)
    #---
    def empty_message(self):
        yield from self._yield_string(DefaultElement.EMPTY_MESSAGE)
    #---
    def stream_response(
        self,
        message: str,
        history: list[list[str]],
        response: StreamingAgentChatResponse,
    ):
        answer = []
        for text in response.response_gen:
            answer.append(text)
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                history + [{"role": "user", "content": message}, {"role": "assistant", "content": "".join(answer)}],
                DefaultElement.ANSWERING_STATUS,
            )
        yield (
            DefaultElement.DEFAULT_MESSAGE,
            history + [{"role": "user", "content": message}, {"role": "assistant", "content": "".join(answer)}],
            DefaultElement.COMPLETED_STATUS,
        )
#------------------------------------------------------------------------------
class LocalChatbotUI:
    def __init__(
        self,
        pipeline: LocalRAGPipeline,
        logger: Logger,
        data_dir: str = "data/data",
        avatar_images: list[str] = ["./assets/user.png", "./assets/bot.png"],
    ):
        self._pipeline = pipeline
        self._logger = logger
        self._data_dir = os.path.join(os.getcwd(), data_dir)
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir, exist_ok=True)
        self._avatar_images = [
            os.path.join(os.getcwd(), image) for image in avatar_images
        ]
        self._variant = "panel"
        self._llm_response = LLMResponse()
    #---
    def _get_response(
        self,
        chat_mode: str,
        message: dict[str, str],
        chatbot: list[dict[str, str]],
        progress=gr.Progress(track_tqdm=True),
    ):
        if self._pipeline.get_model_name() in [None, ""]:
            for m in self._llm_response.set_model():
                yield m
        elif message["text"] in [None, ""]:
            for m in self._llm_response.empty_message():
                yield m
        else:
            console = sys.stdout
            sys.stdout = self._logger
            response = self._pipeline.query(chat_mode, message["text"], chatbot)
            for m in self._llm_response.stream_response(
                message["text"], chatbot, response
            ):
                yield m
            sys.stdout = console
    #---
    def _pull_model(self, model: str, progress=gr.Progress(track_tqdm=True)):
        if (model not in ["gpt-3.5-turbo", "gpt-4"]) and not (
            self._pipeline.check_exist(model)
        ):
            response = self._pipeline.pull_model(model)
            if response.status_code == 200:
                gr.Info(f"Pulling {model}!")
                for data in response.iter_lines(chunk_size=1):
                    data = json.loads(data)
                    if "completed" in data.keys() and "total" in data.keys():
                        progress(data["completed"] / data["total"], desc="Downloading")
                    else:
                        progress(0.0)
            else:
                gr.Warning(f"Model {model} doesn't exist!")
                return (
                    DefaultElement.DEFAULT_MESSAGE,
                    DefaultElement.DEFAULT_HISTORY,
                    DefaultElement.PULL_MODEL_FAIL_STATUS,
                    DefaultElement.DEFAULT_MODEL,
                )
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.PULL_MODEL_SCUCCESS_STATUS,
            model,
        )
    #---
    def _change_model(self, model: str):
        if model not in [None, ""]:
            self._pipeline.set_model_name(model)
            self._pipeline.set_model()
            self._pipeline.set_engine()
            gr.Info(f"Change model to {model}!")
        return DefaultElement.DEFAULT_STATUS
    #---
    def _change_topic(self, topic: str):
        if topic not in [None, ""]:
            self._pipeline.switch_topic(topic)
            gr.Info(f"Switched to topic: {topic}!")
            # Reset UI state
            return (
                DefaultElement.DEFAULT_MESSAGE,
                DefaultElement.DEFAULT_HISTORY,
                DefaultElement.DEFAULT_DOCUMENT,
                DefaultElement.DEFAULT_STATUS,
                self._pipeline.get_system_prompt(),
                gr.update(choices=self._pipeline.get_topics(), value=topic) # Correctly update choices and value
            )
        return (
            gr.update(), gr.update(), gr.update(), 
            DefaultElement.DEFAULT_STATUS, gr.update(), gr.update()
        )
    #---
    def _upload_document(self, document: list[str], list_files: list[str] | dict):
        if document in [None, []]:
            if isinstance(list_files, list):
                return (list_files, DefaultElement.DEFAULT_DOCUMENT)
            else:
                if list_files.get("files", None):
                    return list_files.get("files")
                return document
        else:
            if isinstance(list_files, list):
                return (document + list_files, DefaultElement.DEFAULT_DOCUMENT)
            else:
                if list_files.get("files", None):
                    return document + list_files.get("files")
                return document
    #---
    def _reset_document(self):
        self._pipeline.reset_documents()
        gr.Info("Reset all documents!")
        return (
            DefaultElement.DEFAULT_DOCUMENT,
            gr.update(visible=False),
            gr.update(visible=False),
        )
    #---
    def _show_document_btn(self, document: list[str]):
        visible = False if document in [None, []] else True
        return (gr.update(visible=visible), gr.update(visible=visible))
    #---
    def _processing_document(
        self, document: list[str], progress=gr.Progress(track_tqdm=True)
    ):
        document = document or []
        self._pipeline.store_nodes(input_files=document)
        self._pipeline.set_chat_mode()
        gr.Info("Processing Completed!")
        return (self._pipeline.get_system_prompt(), DefaultElement.COMPLETED_STATUS)
    #---
    def _change_system_prompt(self, sys_prompt: str):
        self._pipeline.set_system_prompt(sys_prompt)
        self._pipeline.set_chat_mode()
        gr.Info("System prompt updated!")
    #---
    def _change_language(self, language: str):
        self._pipeline.set_language(language)
        # Direktaufruf der Funktion aus qa_prompt.py
        new_prompt = get_system_prompt(language)
        self._pipeline.set_system_prompt(new_prompt)
        self._pipeline.set_chat_mode()
        gr.Info(f"Change language to {language}")
        return new_prompt
    #---
    def _delete_database_action(self, scope: str):
        entire_db = (scope == "Entire Database")
        self._pipeline.delete_database(entire_db=entire_db)
        gr.Info(f"{scope} deleted and reset!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_DOCUMENT,
            DefaultElement.DEFAULT_STATUS,
            gr.update(visible=False), # Hide confirm button
            gr.update(value=[]), # Clear documents list in UI
            gr.update(choices=self._pipeline.get_topics(), value=self._pipeline.get_current_topic()) # Update topic dropdown
        )
    #---
    def _undo_chat(self, history: list[list[str, str]]):
        if len(history) > 0:
            if history[-1]["role"] == "assistant":
                history.pop(-1)
            if len(history) > 0 and history[-1]["role"] == "user":
                history.pop(-1)
            return history
        return DefaultElement.DEFAULT_HISTORY
    #---
    def _reset_chat(self):
        self._pipeline.reset_conversation()
        gr.Info("Reset chat!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_DOCUMENT,
            DefaultElement.DEFAULT_STATUS,
        )
    #---
    def _clear_chat(self):
        self._pipeline.clear_conversation()
        gr.Info("Clear chat!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_STATUS,
        )
    #---
    def _show_hide_setting(self, state):
        state = not state
        label = "Hide Setting" if state else "Show Setting"
        return (label, gr.update(visible=state), state)
    #---
    def _welcome(self):
        for m in self._llm_response.welcome():
            yield m
    #---
    def _update_model_list(self):
        models = self._pipeline.get_installed_models()
        # Filter out embedding models to keep the list clean for LLMs
        models = [m for m in models if "embed" not in m.lower()]
        print(f"DEBUG: Models fetched from API (filtered): {models}")
        current_model = self._pipeline.get_model_name()
        new_value = current_model if current_model in models else (models[0] if models else None)
        return gr.update(choices=models, value=new_value)

    # ------------------------------------------------------------------------------
    def build(self):
        installed_models = self._pipeline.get_installed_models()
        llm_models = [m for m in installed_models if "embed" not in m.lower()]
        current_model = self._pipeline.get_model_name()
        default_model = current_model if current_model in llm_models else (llm_models[0] if llm_models else None)
        with gr.Blocks(
            theme=gr.themes.Soft(primary_hue="slate"),
            js=JS_LIGHT_THEME,
            css=CSS
        ) as demo:
            gr.Markdown("## Local RAG Chatbot 🤖")
            with gr.Tab("Interface"):
                sidebar_state = gr.State(True)
                with gr.Row(variant=self._variant, equal_height=False):
                    with gr.Column(
                        variant=self._variant, scale=10, visible=sidebar_state.value
                    ) as setting:
                        with gr.Column():
                            status = gr.Textbox(
                                label="Status", value="Ready!", interactive=False
                            )
                            language = gr.Radio(
                                label="Language",
                                choices=["vi", "eng", "ger"],
                                value="ger",
                                interactive=True,
                            )
                            model = gr.Dropdown(
                                label="Choose Model:",
                                choices=llm_models,
                                value=default_model,
                                interactive=True,
                                allow_custom_value=True,
                                filterable=True,
                            )
                            refresh_models_btn = gr.Button(
                                value="Refresh Models", min_width=50
                            )
                            with gr.Row():
                                pull_btn = gr.Button(
                                    value="Pull Model", visible=False, min_width=50
                                )
                                cancel_btn = gr.Button(
                                    value="Cancel", visible=False, min_width=50
                                )                            
                            topic = gr.Dropdown(
                                label="Choose Topic:",
                                choices=self._pipeline.get_topics(),
                                value=self._pipeline.get_current_topic(),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            documents = gr.Files(
                                label="Add Documents",
                                value=[],
                                file_types=[".txt", ".pdf", ".csv"],
                                file_count="multiple",
                                height=150,
                                interactive=True,
                            )
                            with gr.Row():
                                upload_doc_btn = gr.UploadButton(
                                    label="Upload",
                                    value=[],
                                    file_types=[".txt", ".pdf", ".csv"],
                                    file_count="multiple",
                                    min_width=20,
                                    visible=False,
                                )
                                reset_doc_btn = gr.Button(
                                    "Reset", min_width=20, visible=False
                                )
                    with gr.Column(scale=30, variant=self._variant):
                        chatbot = gr.Chatbot(
                            value=[],
                            height=550,
                            scale=2,
                            show_label=False,
                            avatar_images=self._avatar_images,
                            type="messages",
                        )
                        with gr.Row(variant=self._variant):
                            chat_mode = gr.Dropdown(
                                choices=["chat", "QA"],
                                value="QA",
                                min_width=50,
                                show_label=False,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            message = gr.MultimodalTextbox(
                                value=DefaultElement.DEFAULT_MESSAGE,
                                placeholder="Enter you message:",
                                file_types=[".txt", ".pdf", ".csv"],
                                show_label=False,
                                scale=6,
                                lines=1,
                            )
                        with gr.Row(variant=self._variant):
                            ui_btn = gr.Button(
                                value="Hide Setting"
                                if sidebar_state.value
                                else "Show Setting",
                                min_width=20,
                            )
                            undo_btn = gr.Button(value="Undo", min_width=20)
                            clear_btn = gr.Button(value="Clear", min_width=20)
                            reset_btn = gr.Button(value="Reset", min_width=20)
            with gr.Tab("Setting"):
                with gr.Row(variant=self._variant, equal_height=False):
                    with gr.Column():
                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            value=self._pipeline.get_system_prompt(),
                            interactive=True,
                            lines=10,
                            max_lines=50,
                        )
                        sys_prompt_btn = gr.Button(value="Set System Prompt")
                        #---
                        gr.Markdown("---")
                        gr.Markdown("### ⚠️ Danger Zone")
                        with gr.Row():
                            deletion_scope = gr.Radio(
                                label="Deletion Scope",
                                choices=["Current Topic", "Entire Database"],
                                value="Current Topic",
                                interactive=True
                            )
                        with gr.Row():
                            delete_db_btn = gr.Button("Delete Database", variant="stop")
                            confirm_delete_btn = gr.Button("CONFIRM DELETE", variant="primary", visible=False)
            with gr.Tab("Output"):
                with gr.Row(variant=self._variant):
                    log = gr.Code(
                        label="", language="markdown", interactive=False, lines=30
                    )
                    timer = gr.Timer(1)
                    timer.tick(
                        self._logger.read_logs,
                        outputs=[log],
                        show_progress="hidden",
                    )
            # event handler
            clear_btn.click(self._clear_chat, outputs=[message, chatbot, status])
            cancel_btn.click(
                lambda: (gr.update(visible=False), gr.update(visible=False), None),
                outputs=[pull_btn, cancel_btn, model],
            )
            undo_btn.click(self._undo_chat, inputs=[chatbot], outputs=[chatbot])
            reset_btn.click(
                self._reset_chat, outputs=[message, chatbot, documents, status]
            )
            model.change(
                fn=self._change_model,
                inputs=[model],
                outputs=[status]
            )            
            refresh_models_btn.click(
                self._update_model_list,
                outputs=[model]
            )
            pull_btn.click(
                lambda: (gr.update(visible=False), gr.update(visible=False)),
                outputs=[pull_btn, cancel_btn],
            ).then(
                self._pull_model,
                inputs=[model],
                outputs=[message, chatbot, status, model],
            ).then(self._change_model, inputs=[model], outputs=[status]).then(
                self._update_model_list,
                outputs=[model]
            )
            message.submit(
                self._upload_document, inputs=[documents, message], outputs=[documents]
            ).then(
                self._get_response,
                inputs=[chat_mode, message, chatbot],
                outputs=[message, chatbot, status],
            )
            language.change(self._change_language, inputs=[language], outputs=[system_prompt])
            topic.change(
                self._change_topic,
                inputs=[topic],
                outputs=[message, chatbot, documents, status, system_prompt, topic]
            )
            documents.change(
                self._processing_document,
                inputs=[documents],
                outputs=[system_prompt, status],
            ).then(
                self._show_document_btn,
                inputs=[documents],
                outputs=[upload_doc_btn, reset_doc_btn],
            )
            #---
            sys_prompt_btn.click(self._change_system_prompt, inputs=[system_prompt])
            ui_btn.click(
                self._show_hide_setting,
                inputs=[sidebar_state],
                outputs=[ui_btn, setting, sidebar_state],
            )
            upload_doc_btn.upload(
                self._upload_document,
                inputs=[documents, upload_doc_btn],
                outputs=[documents, upload_doc_btn],
            )
            reset_doc_btn.click(
                self._reset_document, outputs=[documents, upload_doc_btn, reset_doc_btn]
            )            
            delete_db_btn.click(
                lambda: gr.update(visible=True),
                outputs=[confirm_delete_btn]
            )            
            confirm_delete_btn.click(
                self._delete_database_action,
                inputs=[deletion_scope],
                outputs=[message, chatbot, documents, status, confirm_delete_btn, upload_doc_btn, topic]
            )            
            demo.load(self._welcome, outputs=[message, chatbot, status]).then(
                fn=self._change_model,
                inputs=[model],
                outputs=[status]
            )
        return demo
