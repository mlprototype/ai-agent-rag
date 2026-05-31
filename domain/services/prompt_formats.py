from pathlib import Path
from typing import Any, Literal

import yaml
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, model_validator


class PromptMessageDocument(BaseModel):
    role: Literal["system", "human", "ai", "messages_placeholder"]
    template: str | None = None
    variable_name: str | None = None

    @model_validator(mode="after")
    def validate_shape(self):
        if self.role == "messages_placeholder":
            if not self.variable_name or self.template is not None:
                raise ValueError("messages_placeholder requires variable_name and forbids template")
        elif not self.template or self.variable_name is not None:
            raise ValueError("chat message requires template and forbids variable_name")
        return self


class PromptDocument(BaseModel):
    version: str
    format: Literal["chat_prompt"] = "chat_prompt"
    messages: list[PromptMessageDocument]


def classify_prompt_error(exc: Exception) -> str:
    error_name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    module_name = exc.__class__.__module__.lower()
    status_code = getattr(getattr(exc, "response", None), "status_code", None)

    if error_name == "langsmithnotfounderror":
        return "prompt_not_found"
    if error_name == "langsmithautherror":
        return "auth_error"
    if error_name == "langsmithratelimiterror":
        return "rate_limited"
    if error_name == "langsmithconnectionerror":
        return "connection_error"
    if error_name == "langsmithrequesttimeout":
        return "timeout"
    if error_name == "langsmithapierror":
        return "hub_server_error"
    if error_name == "langsmithusererror" and "workspace specification" in message:
        return "workspace_required"
    if "workspace specification" in message:
        return "workspace_required"
    if "no prompt owner specified" in message:
        return "prompt_owner_required"
    if error_name == "langsmithusererror":
        return "langsmith_user_error"
    if error_name == "langsmitherror":
        return "langsmith_error"

    if error_name == "filenotfounderror":
        return "prompt_not_found"
    if status_code == 404:
        return "prompt_not_found"
    if status_code in {401, 403}:
        return "auth_error"
    if status_code == 429:
        return "rate_limited"
    if status_code and status_code >= 500:
        return "hub_server_error"
    if "404" in message:
        return "prompt_not_found"
    if "401" in message or "403" in message:
        return "auth_error"
    if "429" in message:
        return "rate_limited"
    if "500" in message or "502" in message or "503" in message or "504" in message:
        return "hub_server_error"
    if "connection" in error_name or "connection" in message:
        return "connection_error"
    if error_name == "validationerror":
        return "schema_validation_error"
    if error_name in {"parsererror", "scannererror", "composererror"} or "yaml" in module_name:
        return "template_parse_error"
    if error_name in {"timeouterror", "timeout"}:
        return "timeout"
    if "invalid_prompt_input" in message or "missing variables" in message or "chatprompttemplate" in message:
        return "template_parse_error"
    if "timeout" in message:
        return "timeout"
    return error_name


def extract_prompt_error_detail(exc: Exception, *, limit: int = 400) -> str:
    detail = str(exc).strip() or repr(exc)
    detail = " ".join(detail.split())
    if len(detail) > limit:
        return detail[: limit - 3] + "..."
    return detail


def validate_prompt_payload(payload: Any) -> PromptDocument:
    return PromptDocument.model_validate(payload)


def load_prompt_document_from_path(path: str | Path) -> PromptDocument:
    prompt_path = Path(path)
    with prompt_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return validate_prompt_payload(payload)


def build_prompt_from_document(document: PromptDocument) -> ChatPromptTemplate:
    messages = []
    for message in document.messages:
        if message.role == "messages_placeholder":
            messages.append(MessagesPlaceholder(message.variable_name))
        else:
            messages.append((message.role, message.template))
    return ChatPromptTemplate.from_messages(messages)


def prompt_to_document(prompt: Any, *, version: str) -> PromptDocument:
    if not isinstance(prompt, ChatPromptTemplate):
        raise ValueError(f"unsupported_prompt_type:{type(prompt).__name__}")

    messages: list[PromptMessageDocument] = []
    for message in prompt.messages:
        if isinstance(message, MessagesPlaceholder):
            messages.append(
                PromptMessageDocument(
                    role="messages_placeholder",
                    variable_name=message.variable_name,
                )
            )
            continue

        if isinstance(message, SystemMessagePromptTemplate):
            role = "system"
        elif isinstance(message, HumanMessagePromptTemplate):
            role = "human"
        elif isinstance(message, AIMessagePromptTemplate):
            role = "ai"
        else:
            raise ValueError(f"unsupported_chat_message_type:{type(message).__name__}")

        template = getattr(getattr(message, "prompt", None), "template", None)
        template_format = getattr(getattr(message, "prompt", None), "template_format", None)
        if not template:
            raise ValueError(f"missing_template:{type(message).__name__}")
        if template_format not in {None, "f-string"}:
            raise ValueError(f"unsupported_template_format:{template_format}")

        messages.append(PromptMessageDocument(role=role, template=template))

    return PromptDocument(version=version, format="chat_prompt", messages=messages)


class _PromptYamlDumper(yaml.SafeDumper):
    pass


def _represent_multiline_str(dumper: yaml.SafeDumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_PromptYamlDumper.add_representer(str, _represent_multiline_str)


def dump_prompt_document(document: PromptDocument) -> str:
    payload = document.model_dump(exclude_none=True)
    return yaml.dump(
        payload,
        Dumper=_PromptYamlDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
