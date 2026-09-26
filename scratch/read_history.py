import json

with open(r'C:/Users/matee/.gemini/antigravity-ide/brain/845c2ca7-9f8c-48b4-b21f-1830af23387f/.system_generated/logs/transcript.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d.get('type') == 'USER_INPUT':
            print(f"=== STEP {d.get('step_index')} ===")
            print(d.get('content'))
