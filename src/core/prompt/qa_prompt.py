def get_context_prompt(language: str) -> str:
    if language == "vi":
        return CONTEXT_PROMPT_VI
    return CONTEXT_PROMPT_EN


def get_system_prompt(language: str, is_rag_prompt: bool = True) -> str:
    if language == "vi":
        return SYSTEM_PROMPT_RAG_VI if is_rag_prompt else SYSTEM_PROMPT_VI
    elif language == "ger":
        return SYSTEM_PROMPT_RAG_DE if is_rag_prompt else SYSTEM_PROMPT_DE
    return SYSTEM_PROMPT_RAG_EN if is_rag_prompt else SYSTEM_PROMPT_EN


SYSTEM_PROMPT_EN = """\
This is a chat between a user and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. \
The assistant should also indicate when the answer cannot be found in the context."""

SYSTEM_PROMPT_RAG_EN = """\
This is a chat between a user and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. \
The assistant should also indicate when the answer cannot be found in the context."""

CONTEXT_PROMPT_EN = """\
Here are the relevant documents for the context:

{context_str}

Instruction: Based on the above documents, provide a detailed answer for the user question below. \
Answer 'don't know' if not present in the document."""

CONDENSED_CONTEXT_PROMPT_EN = """\
Given the following conversation between a user and an AI assistant and a follow up question from user,
rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:\
"""

SYSTEM_PROMPT_DE = """\
Dies ist ein Chat zwischen einem Benutzer und einem Assistenten mit künstlicher Intelligenz. \
Der Assistent gibt hilfreiche, detaillierte und höfliche Antworten auf die Fragen des Benutzers basierend auf dem Kontext. \
Der Assistent sollte zudem darauf hinweisen, wenn die Antwort nicht im Kontext gefunden werden kann."""

SYSTEM_PROMPT_RAG_DE = """\
Dies ist ein Chat zwischen einem Benutzer und einem Assistenten mit künstlicher Intelligenz. \
Der Assistent gibt hilfreiche, detaillierte und höfliche Antworten auf die Fragen des Benutzers basierend auf dem Kontext. \
Der Assistent sollte zudem darauf hinweisen, wenn die Antwort nicht im Kontext gefunden werden kann."""

CONTEXT_PROMPT_DE = """\
Hier sind die relevanten Dokumente für den Kontext:

{context_str}

Anweisung: Beantworte die untenstehende Benutzerfrage detailliert auf Basis der oben genannten Dokumente. \
Antworte mit „Ich weiß es nicht“, falls die Information nicht im Dokument enthalten ist."""

CONDENSED_CONTEXT_PROMPT_DE = """\
Formuliere die folgende Anschlussfrage eines Benutzers in eine eigenständige Frage um, \
basierend auf dem bisherigen Gesprächsverlauf zwischen dem Benutzer und dem KI-Assistenten.

Chat-Verlauf:
{chat_history}
Anschlussfrage: {question}
Eigenständige Frage:\
"""

SYSTEM_PROMPT_VI = """\
Đây là một cuộc trò chuyện giữa người dùng và một trợ lí trí tuệ nhân tạo. \
Trợ lí đưa ra các câu trả lời hữu ích, chi tiết và lịch sự đối với các câu hỏi của người dùng dựa trên bối cảnh. \
Trợ lí cũng nên chỉ ra khi câu trả lời không thể được tìm thấy trong ngữ cảnh."""

SYSTEM_PROMPT_RAG_VI = """\
Đây là một cuộc trò chuyện giữa người dùng và một trợ lí trí tuệ nhân tạo. \
Trợ lí đưa ra các câu trả lời hữu ích, chi tiết và lịch sự đối với các câu hỏi của người dùng dựa trên bối cảnh. \
Trợ lí cũng nên chỉ ra khi câu trả lời không thể được tìm thấy trong ngữ cảnh."""

CONTEXT_PROMPT_VI = """\
Dưới đây là các tài liệu liên quan cho ngữ cảnh:

{context_str}

Hướng dẫn: Dựa trên các tài liệu trên, cung cấp một câu trả lời chi tiết cho câu hỏi của người dùng dưới đây. \
Trả lời 'không biết' nếu không có trong tài liệu."""

CONDENSED_CONTEXT_PROMPT_VI = """\
Cho cuộc trò chuyện sau giữa một người dùng và một trợ lí trí tuệ nhân tạo và một câu hỏi tiếp theo từ người dùng,
đổi lại câu hỏi tiếp theo để là một câu hỏi độc lập.

Lịch sử Trò chuyện:
{chat_history}
Đầu vào Tiếp Theo: {question}
Câu hỏi độc lập:\
"""
