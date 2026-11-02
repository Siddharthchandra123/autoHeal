"""Meta Muse Spark advisor. Never grants itself execution permission."""
import json
import os


def propose(diagnosis: str, history: list[dict], current: dict, model: str) -> dict | None:
    """Send only small metric/config summaries, not datasets or secret values."""
    key = os.environ.get('MODEL_API_KEY')
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(base_url='https://api.meta.ai/v1', api_key=key, timeout=25.0, max_retries=1)
        payload = {
            'diagnosis': diagnosis,
            'recent_metrics': history[-8:],
            'current_hyperparameters': current,
            'allowed_keys': ['learning_rate', 'weight_decay', 'lora_dropout', 'num_train_epochs', 'lora_r', 'gradient_accumulation_steps'],
        }
        answer = client.chat.completions.create(
            model=model,
            messages=[
                {'role':'developer','content': 'You are a constrained ML hyperparameter diagnostic advisor. Return ONLY a JSON object with one key called patch, mapping allowed hyperparameter names to numbers. Do not request file edits, tools, commands, or dataset access. Keep changes modest.'},
                {'role':'user','content': json.dumps(payload)}],
        )
        raw = (answer.choices[0].message.content or '').strip()
        # Handle optional code fence; validation occurs independently in policy.py.
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        obj = json.loads(raw)
        return obj.get('patch') if isinstance(obj, dict) else None
    except Exception as exc:
        print(f'[Meta agent unavailable] {type(exc).__name__}: falling back to deterministic policy')
        return None
