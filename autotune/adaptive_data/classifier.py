def classify_overflow(profile):
    if profile.get("fits"):
        return "fits"
    limit = int(profile["max_length"])
    assistant = int(profile.get("assistant_content_tokens", 0))
    context = int(profile.get("non_assistant_content_tokens", 0))
    if assistant > limit * .5 and context > limit * .5:
        return "mixed_overflow"
    if assistant > limit * .5:
        return "response_overflow"
    return "context_overflow"
