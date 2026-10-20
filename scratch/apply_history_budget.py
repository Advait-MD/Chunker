with open('face.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''            history_to_use = session_history if session_history is not None else self.conversation_history
            if history_to_use:
                prompt_parts.append("\\nRecent chat:")
                if any('role' in m for m in history_to_use if isinstance(m, dict)):
                    for msg in history_to_use[-4:]:
                        role = "User" if msg.get('role') == 'user' else "APEX"
                        c = msg.get('content', '').strip()
                        if c and c != context['query']:
                            prompt_parts.append(f"{role}: {c[:150]}")
                else:
                    for turn in history_to_use[-2:]:
                        q = turn.get('question', '')
                        a = turn.get('answer', '')
                        if q and a:
                            prompt_parts.append(f"User: {q}")
                            prompt_parts.append(f"APEX: {a[:150]}")'''

replacement = '''            history_to_use = session_history if session_history is not None else self.conversation_history
            if history_to_use:
                prompt_parts.append("\\nRecent chat:")
                if any('role' in m for m in history_to_use if isinstance(m, dict)):
                    for msg in history_to_use[-2:]:
                        role = "User" if msg.get('role') == 'user' else "APEX"
                        c = msg.get('content', '').strip()
                        if c and c != context['query']:
                            prompt_parts.append(f"{role}: {c[:75]}")
                else:
                    for turn in history_to_use[-1:]:
                        q = turn.get('question', '')
                        a = turn.get('answer', '')
                        if q and a:
                            prompt_parts.append(f"User: {q[:75]}")
                            prompt_parts.append(f"APEX: {a[:75]}")'''

if target in text:
    text = text.replace(target, replacement, 1)
    with open('face.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Successfully budgeted multi-turn conversation history in face.py!")
else:
    print("Target block not found in face.py!")
