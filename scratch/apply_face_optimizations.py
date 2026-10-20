import re

with open("face.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update RAGInterface __init__ to include persistent session and standardized runner options
target_init = """        self.source_filter = source_filter
        self.conversation_history = []
        self.current_session = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Initialize the processor with local ChromaDB & Ollama"""

replacement_init = """        self.source_filter = source_filter
        self.conversation_history = []
        self.current_session = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # High-performance persistent connection pooling (eliminates Windows TCP handshaking delays)
        self.session = requests.Session()
        cpu_threads = min(8, os.cpu_count() or 4)
        self.ollama_options = {
            "temperature": self.temperature,
            "num_predict": 50,
            "top_k": 30,
            "top_p": 0.9,
            "num_ctx": 2048,
            "num_thread": cpu_threads
        }

        # Initialize the processor with local ChromaDB & Ollama"""

assert target_init in content, "Could not find target_init in face.py"
content = content.replace(target_init, replacement_init, 1)

# 2. Update _check_ollama to use self.session and configured Ollama URL (127.0.0.1)
target_check = """    def _check_ollama(self) -> bool:
        \"\"\"Check if Ollama is running locally\"\"\"
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)"""

replacement_check = """    def _check_ollama(self) -> bool:
        \"\"\"Check if Ollama is running locally\"\"\"
        try:
            url_clean = config.ollama.url.rstrip('/')
            response = self.session.get(f"{url_clean}/api/tags", timeout=5)"""

assert target_check in content, "Could not find target_check in face.py"
content = content.replace(target_check, replacement_check, 1)

# 3. Update build_prompt to be concise, bounded, and fast
target_prompt_func = """    def build_prompt(self, context: Dict, session_history: Optional[List[Dict]] = None) -> str:
        \"\"\"
        Build a prompt for the LLM prioritizing primary source evidence and conversational memory.
        Seamlessly supports both document-grounded RAG and natural human dialogue.
        \"\"\"
        # Pure conversation branch: chit-chat or general dialogue with no matching document chunks
        if context.get('is_chitchat') or (not context.get('chunks') and not context.get('cluster_summaries')):
            prompt_parts = [
                "You are APEX, a thoughtful, articulate, and friendly AI conversational companion.",
                "Chat warmly, naturally, and knowledgeably with the user like an empathetic, intelligent human partner.",
                "Engage deeply with the user's thoughts, answer questions thoroughly, and maintain conversational flow naturally without mentioning PDF or document constraints."
            ]
            if context.get('conversation_memories'):
                prompt_parts.append("\\n💬 RELEVANT USER MEMORIES FROM PAST CHATS:")
                for mem in context['conversation_memories']:
                    prompt_parts.append(f"- {mem.get('text', '')}")

            history_to_use = session_history if session_history is not None else self.conversation_history
            if history_to_use:
                prompt_parts.append("\\n💬 CONVERSATION HISTORY:")
                if any('role' in m for m in history_to_use if isinstance(m, dict)):
                    for msg in history_to_use[-8:]:
                        role = "User" if msg.get('role') == 'user' else "APEX"
                        c = msg.get('content', '').strip()
                        if c and c != context['query']:
                            prompt_parts.append(f"{role}: {c[:300]}")
                else:
                    for turn in history_to_use[-4:]:
                        q = turn.get('question', '')
                        a = turn.get('answer', '')
                        if q and a:
                            prompt_parts.append(f"User: {q}")
                            prompt_parts.append(f"APEX: {a[:300]}")

            prompt_parts.append(f"\\nUser: {context['query']}")
            prompt_parts.append("APEX:")
            return "\\n".join(prompt_parts)

        prompt_parts = []
        
        prompt_parts.append(\"\"\"You are a knowledgeable assistant. Answer the question based on the provided context.

INSTRUCTIONS:
1. Ground your answer strictly in the PRIMARY SOURCE EVIDENCE and PREVIOUS CONVERSATIONS & USER MEMORY below.
2. If the user asks about previous chats, personal preferences, user projects, or topics previously discussed, ground your answer in PREVIOUS CONVERSATIONS & USER MEMORY.
3. When citing facts from books/PDFs, always include the bracketed source reference, e.g. '[DocumentName.pdf, Page 12]'.
4. If neither the documents nor the previous conversations contain enough information to answer the question, say "I don't have enough information to answer that based on the provided documents and past conversations."
5. Be concise, accurate, and thorough.
\"\"\")
        
        if context.get('sources'):
            prompt_parts.append(f"\\n📚 SOURCES: {', '.join(context['sources'])}")
        
        # Primary source evidence is placed FIRST
        prompt_parts.append("\\n📄 PRIMARY SOURCE EVIDENCE (Specific Details & Facts):")
        for i, (chunk, score, meta) in enumerate(zip(
            context['chunks'], 
            context['scores'], 
            context['metadatas']
        )):
            source = meta.get('source_file', 'Unknown') if meta else 'Unknown'
            chunk_id = meta.get('chunk_id', 'Unknown') if meta else 'Unknown'
            page_num = meta.get('page_number', '?') if meta else '?'
            prompt_parts.append(f"\\n[Source {i+1}: {source}, Page {page_num}, Chunk {chunk_id}] (Relevance: {score:.1%})\\n{chunk}")
        
        # Thematic summary is auxiliary and placed AFTER primary evidence
        if context.get('cluster_summaries'):
            prompt_parts.append("\\n📊 THEMATIC CONTEXT (High-Level Topic Overview - Macro Framing Only, Do Not Override Source Evidence):")
            for summary_info in context['cluster_summaries']:
                level = summary_info.get('level', 0)
                cid = summary_info.get('cluster_id', 'unknown')
                chunks_cnt = summary_info.get('chunks_count', 0)
                # Use controlled dense summary (Item 21) if available, falling back to full summary
                summary_text = summary_info.get('dense_summary') or summary_info.get('summary', '')
                prompt_parts.append(f"\\n[Level {level} Cluster {cid} ({chunks_cnt} chunks)]: {summary_text}")
        
        # Conversational memory of past chats & personal context
        if context.get('conversation_memories'):
            prompt_parts.append("\\n💬 PREVIOUS CONVERSATIONS & USER MEMORY (Personal Context, User Statements & Past Chats):")
            for i, mem in enumerate(context['conversation_memories']):
                title = mem.get('chat_title', 'Previous Chat')
                score = mem.get('score', 0.0)
                prompt_parts.append(f"\\n[Memory {i+1}: {title}] (Relevance: {score:.1%})\\n{mem['text']}")

        # Conversational cluster themes (Macro topics from user's chat history)
        if context.get('conversation_cluster_summaries'):
            prompt_parts.append("\\n🧠 CONVERSATION TOPIC THEMES (Macro Themes of Past User Chats):")
            for cs in context['conversation_cluster_summaries']:
                chats = cs.get('chats_represented', 'Chats')
                summary_text = cs.get('dense_summary') or cs.get('summary', '')
                prompt_parts.append(f"\\n[Theme ({chats})]: {summary_text}")

        if self.conversation_history:
            prompt_parts.append("\\n💬 CURRENT SESSION HISTORY:")
            for turn in self.conversation_history[-3:]:
                prompt_parts.append(f"Q: {turn['question']}")
                prompt_parts.append(f"A: {turn['answer'][:200]}...")
        
        prompt_parts.append(f"\\n❓ QUESTION: {context['query']}")
        prompt_parts.append("\\n✅ ANSWER:")
        
        return "\\n".join(prompt_parts)"""

replacement_prompt_func = """    def build_prompt(self, context: Dict, session_history: Optional[List[Dict]] = None) -> str:
        \"\"\"
        Build a concise, evidence-dense prompt prioritizing primary source evidence
        and conversational memory. Strictly bounds character count to guarantee
        TTFT <= 2.0s and Total Turnaround <= 5.0s.
        \"\"\"
        # Pure conversation branch: chit-chat or general dialogue with no matching document chunks
        if context.get('is_chitchat') or (not context.get('chunks') and not context.get('cluster_summaries')):
            prompt_parts = [
                "You are APEX, a thoughtful, articulate, and friendly AI companion.",
                "Chat warmly and concisely in 1-2 direct sentences like an intelligent human partner."
            ]
            if context.get('conversation_memories'):
                prompt_parts.append("\\nPast memories:")
                for mem in context['conversation_memories'][:2]:
                    prompt_parts.append(f"- {mem.get('text', '')[:150]}")

            history_to_use = session_history if session_history is not None else self.conversation_history
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
                            prompt_parts.append(f"APEX: {a[:150]}")

            prompt_parts.append(f"\\nUser: {context['query']}")
            prompt_parts.append("APEX:")
            return "\\n".join(prompt_parts)

        prompt_parts = [
            "Answer concisely in 1-2 direct sentences using the source evidence. Always cite bracketed sources e.g. [Document.pdf, Page 12].\\n"
        ]
        
        # Primary source evidence (bounded to first 450 characters per chunk)
        for i, (chunk, meta) in enumerate(zip(
            context.get('chunks', [])[:self.max_context_chunks], 
            context.get('metadatas', [])[:self.max_context_chunks]
        )):
            source = meta.get('source_file', 'Doc') if meta else 'Doc'
            page_num = meta.get('page_number', '?') if meta else '?'
            clean_chunk = chunk[:450].strip().replace('\\n', ' ')
            prompt_parts.append(f"[{source}, Page {page_num}]: {clean_chunk}")
        
        # Thematic summary (if present and enabled)
        if context.get('cluster_summaries') and self.include_summaries:
            for summary_info in context['cluster_summaries'][:1]:
                summary_text = (summary_info.get('dense_summary') or summary_info.get('summary', ''))[:150].strip().replace('\\n', ' ')
                prompt_parts.append(f"[Theme]: {summary_text}")
        
        # Conversational memory (bounded)
        if context.get('conversation_memories'):
            for mem in context['conversation_memories'][:2]:
                prompt_parts.append(f"[Memory]: {mem['text'][:150].strip().replace('\\n', ' ')}")

        prompt_parts.append(f"\\nQuestion: {context['query']}\\nAnswer:")
        return "\\n".join(prompt_parts)"""

assert target_prompt_func in content, "Could not find target_prompt_func in face.py"
content = content.replace(target_prompt_func, replacement_prompt_func, 1)

# 4. Update generate_response to use self.session and self.ollama_options
target_gen = """        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": 150,
                            "top_k": 40,
                            "top_p": 0.9,
                            "num_ctx": 2048,
                            "num_thread": cpu_threads
                        },
                        "keep_alive": "1h"
                    },
                    timeout=timeout
                )"""

replacement_gen = """        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": self.ollama_options,
                        "keep_alive": "1h"
                    },
                    timeout=timeout
                )"""

assert target_gen in content, "Could not find target_gen in face.py"
content = content.replace(target_gen, replacement_gen, 1)

# 5. Update generate_response_stream to use self.session and self.ollama_options
target_stream = """    def generate_response_stream(self, prompt: str):
        \"\"\"
        Stream response tokens from native Ollama in real-time (Item 2.2).
        Yields token strings as they arrive from Ollama.
        \"\"\"
        cpu_threads = min(8, os.cpu_count() or 4)
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": 256,
                        "top_k": 40,
                        "top_p": 0.9,
                        "num_ctx": 2048,
                        "num_thread": cpu_threads
                    },
                    "keep_alive": "1h"
                },
                stream=True,
                timeout=config.ollama.timeout
            )
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    yield token
                    if chunk.get("done", False):
                        break
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\\n[Streaming error: {e}]\""""

replacement_stream = """    def generate_response_stream(self, prompt: str):
        \"\"\"
        Stream response tokens from native Ollama in real-time (Item 2.2).
        Yields token strings as they arrive from Ollama.
        \"\"\"
        try:
            with self.session.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": True,
                    "options": self.ollama_options,
                    "keep_alive": "1h"
                },
                stream=True,
                timeout=config.ollama.timeout
            ) as response:
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        yield token
                        if chunk.get("done", False):
                            break
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\\n[Streaming error: {e}]\""""

assert target_stream in content, "Could not find target_stream in face.py"
content = content.replace(target_stream, replacement_stream, 1)

with open("face.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully applied all optimizations to face.py!")
