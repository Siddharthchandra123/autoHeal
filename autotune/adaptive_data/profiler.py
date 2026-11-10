from __future__ import annotations

def profile_messages(tokenizer, messages, max_length, chat_template_kwargs=None):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=False, **(chat_template_kwargs or {}))
    ids = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
    roles = {}
    for item in messages:
        if not isinstance(item, dict) or not isinstance(item.get("content"), str):
            raise ValueError("each message must contain string content")
        role = str(item.get("role", "unknown"))
        roles[role] = roles.get(role, 0) + len(tokenizer(
            item["content"], add_special_tokens=False, truncation=False)["input_ids"])
    assistant = roles.get("assistant", 0)
    context = sum(v for k, v in roles.items() if k != "assistant")
    total = len(ids)
    return {"total_tokens": total, "max_length": max_length, "fits": total <= max_length,
            "assistant_content_tokens": assistant, "non_assistant_content_tokens": context,
            "overflow_tokens": max(0, total-max_length), "rendered_chars": len(rendered)}
