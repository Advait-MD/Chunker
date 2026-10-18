import requests
import pandas as pd
from typing import List, Dict, Optional
import time
from datetime import datetime
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import backend from main.py
try:
    from main import HierarchicalClusterSummarizer
except ImportError:
    print("❌ Error: Could not import HierarchicalClusterSummarizer from main.py")
    print("   Make sure main.py is in the same directory")
    sys.exit(1)


class RAGInterface:
    """
    Conversational RAG Interface using main.py for retrieval and Ollama for generation
    Runs locally using ChromaDB PersistentClient and native Ollama
    """
    
    def __init__(self, 
                 ollama_model: str = "gemma3:1b",
                 ollama_embedding_model: str = "nomic-embed-text",
                 ollama_url: str = "http://localhost:11434",
                 chroma_path: Optional[str] = "./hierarchical_chroma_db",
                 chroma_host: Optional[str] = None,
                 chroma_port: Optional[int] = None,
                 max_context_chunks: int = 5,
                 include_summaries: bool = True,
                 temperature: float = 0.3,
                 verbose: bool = True):
        """
        Initialize the RAG Interface with local services
        
        Args:
            ollama_model: Ollama model name for generation
            ollama_embedding_model: Ollama model for embeddings
            ollama_url: Ollama API endpoint (base URL)
            chroma_path: Path to local ChromaDB directory
            chroma_host: ChromaDB host (optional, for remote server)
            chroma_port: ChromaDB port (optional, for remote server)
            max_context_chunks: Number of chunks to retrieve
            include_summaries: Whether to include cluster summaries
            temperature: LLM temperature (0-1)
            verbose: Print detailed information
        """
        self.verbose = verbose
        
        # Initialize the processor with local ChromaDB & Ollama
        self.processor = HierarchicalClusterSummarizer(
            ollama_model=ollama_model,
            ollama_embedding_model=ollama_embedding_model,
            ollama_url=ollama_url,
            chroma_path=chroma_path,
            chroma_host=chroma_host,
            chroma_port=chroma_port
        )
        
        self.ollama_model = ollama_model
        self.ollama_url = f"{ollama_url}/api/generate"  # Full API endpoint
        self.max_context_chunks = max_context_chunks
        self.include_summaries = include_summaries
        self.temperature = temperature
        self.conversation_history = []
        self.current_session = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check connections
        self._check_ollama()
        self._show_database_info()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running locally"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                if any(self.ollama_model in name for name in model_names):
                    if self.verbose:
                        print(f"✅ Ollama is running with model: {self.ollama_model}")
                    return True
                else:
                    print(f"⚠️ Model {self.ollama_model} not found. Run: ollama pull {self.ollama_model}")
                    return False
            return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("   Please start Ollama with: ollama serve (or launch Ollama from the Start Menu)")
            return False
    
    def _show_database_info(self):
        """Show information about the database from local ChromaDB"""
        try:
            total_chunks = self.processor.chunks_collection.count()
            total_summaries = self.processor.summaries_collection.count()
            
            if self.verbose:
                print(f"📚 ChromaDB (local): {total_chunks} chunks, {total_summaries} summaries")
                
                # Show source distribution
                if total_chunks > 0:
                    result = self.processor.chunks_collection.get(include=["metadatas"], limit=10000)
                    sources = {}
                    for meta in result['metadatas']:
                        source = meta.get('source_file', 'Unknown')
                        sources[source] = sources.get(source, 0) + 1
                    
                    print(f"📄 Sources: {len(sources)} document(s)")
                    for source, count in sources.items():
                        print(f"   • {source}: {count} chunks")
                else:
                    print("   ⚠️ Database is empty! Run main.py first to add documents.")
                    
        except Exception as e:
            print(f"⚠️ Could not get database info: {e}")
    
    def retrieve_context(self, query: str) -> Dict:
        """
        Retrieve relevant context from local ChromaDB using HNSW search
        
        Args:
            query: User question
            
        Returns:
            Dictionary with retrieved chunks, metadata, and scores
        """
        if not query or not query.strip():
            return {'success': False, 'message': 'Empty or blank query provided'}

        if self.verbose:
            print(f"\n🔍 Retrieving context for: {query[:80]}...")
        
        try:
            # Check if collection is empty
            if self.processor.chunks_collection.count() == 0:
                return {'success': False, 'message': 'Database is empty. Please ingest documents with main.py first.'}

            # Query chunks_collection for HNSW search
            results = self.processor.chunks_collection.query(
                query_texts=[query.strip()],
                n_results=self.max_context_chunks,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {'success': False, 'message': 'No relevant documents found'}
            
            retrieved_chunks = results['documents'][0]
            retrieved_metadatas = results['metadatas'][0]
            retrieved_distances = results['distances'][0]
            
            # Calculate relevance scores (cosine distance to similarity, clipped [0, 1])
            relevance_scores = [max(0.0, min(1.0, 1.0 - float(d))) for d in retrieved_distances]
            low_confidence = all(s < 0.25 for s in relevance_scores) if relevance_scores else True
            
            # Separately retrieve thematic cluster summaries if requested (Collection Separation)
            cluster_summaries = []
            if self.include_summaries and self.processor.summaries_collection.count() > 0:
                try:
                    summary_results = self.processor.summaries_collection.query(
                        query_texts=[query.strip()],
                        n_results=1,
                        include=["documents", "metadatas", "distances"]
                    )
                    if summary_results and summary_results.get('documents') and summary_results['documents'][0]:
                        for s_doc, s_meta, s_dist in zip(
                            summary_results['documents'][0],
                            summary_results['metadatas'][0],
                            summary_results['distances'][0]
                        ):
                            s_score = max(0.0, min(1.0, 1.0 - float(s_dist)))
                            # Only include summary if it has meaningful topical relevance (>= 25%)
                            if s_score >= 0.25:
                                cluster_summaries.append({
                                    'cluster_id': s_meta.get('cluster_id', 'unknown'),
                                    'level': s_meta.get('level', 0),
                                    'chunks_count': s_meta.get('chunks_count', 0),
                                    'score': s_score,
                                    'sources': s_meta.get('source_files', []),
                                    'summary': s_doc[:600]
                                })
                except Exception as e:
                    if self.verbose:
                        print(f"  Could not fetch cluster summary: {e}")
            
            # Extract unique sources
            sources = []
            for meta in retrieved_metadatas:
                if meta and 'source_file' in meta:
                    sources.append(meta['source_file'])
            sources = list(set(sources))
            
            return {
                'success': True,
                'chunks': retrieved_chunks,
                'metadatas': retrieved_metadatas,
                'scores': relevance_scores,
                'cluster_summaries': cluster_summaries,
                'sources': sources,
                'query': query,
                'low_confidence': low_confidence
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Error retrieving context: {str(e)}'}
    
    def build_prompt(self, context: Dict) -> str:
        """
        Build a prompt for the LLM using retrieved context
        
        Args:
            context: Retrieved context dictionary from retrieve_context()
            
        Returns:
            Formatted prompt
        """
        prompt_parts = []
        
        prompt_parts.append("""You are a knowledgeable assistant. Answer the question based ONLY on the provided context.

INSTRUCTIONS:
- Ground your answer in the PRIMARY SOURCE EVIDENCE below.
- Use THEMATIC BACKGROUND SUMMARY only for high-level macro context.
- If the context doesn't contain the answer, say "I don't have enough information to answer that"
- Be concise, accurate, and thorough.
- When quoting or citing, mention the document name.
- Use a helpful, professional tone.
""")
        
        if context.get('sources'):
            prompt_parts.append(f"\n📚 SOURCES: {', '.join(context['sources'])}")
        
        if context.get('cluster_summaries'):
            prompt_parts.append("\n📊 THEMATIC BACKGROUND SUMMARY (Macro Context):")
            for summary_info in context['cluster_summaries']:
                level = summary_info.get('level', 0)
                cid = summary_info.get('cluster_id', 'unknown')
                chunks_cnt = summary_info.get('chunks_count', 0)
                prompt_parts.append(f"\n[Level {level} Topic {cid} - {chunks_cnt} chunks]:")
                prompt_parts.append(f"{summary_info['summary']}")
        
        prompt_parts.append("\n📄 PRIMARY SOURCE EVIDENCE (Specific Details):")
        for i, (chunk, score, meta) in enumerate(zip(
            context['chunks'], 
            context['scores'], 
            context['metadatas']
        )):
            source = meta.get('source_file', 'Unknown') if meta else 'Unknown'
            chunk_id = meta.get('chunk_id', 'Unknown') if meta else 'Unknown'
            prompt_parts.append(f"\n[{i+1}] (Source: {source}, Chunk: {chunk_id}, Relevance: {score:.2%})")
            prompt_parts.append(f"{chunk}")
        
        if self.conversation_history:
            prompt_parts.append("\n💬 CONVERSATION HISTORY:")
            for turn in self.conversation_history[-3:]:
                prompt_parts.append(f"Q: {turn['question']}")
                prompt_parts.append(f"A: {turn['answer'][:200]}...")
        
        prompt_parts.append(f"\n❓ QUESTION: {context['query']}")
        prompt_parts.append("\n✅ ANSWER:")
        
        return "\n".join(prompt_parts)
    
    def generate_response(self, prompt: str) -> Dict:
        """
        Generate response using native Ollama LLM
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Dictionary with response and metadata
        """
        if self.verbose:
            print(" Generating response...")
            
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": 300,
                        "top_k": 40,
                        "top_p": 0.9
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                return {
                    'success': True,
                    'answer': answer,
                    'tokens': len(answer.split()),
                    'model': self.ollama_model
                }
            else:
                return {
                    'success': False,
                    'error': f"Ollama API returned status: {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            return {'success': False, 'error': "Ollama request timed out after 90s. The local model may still be loading."}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': "Cannot connect to Ollama on http://localhost:11434. Make sure 'ollama serve' is running."}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def query(self, question: str) -> Dict:
        """
        Complete RAG query: Retrieve using main.py, Augment, Generate
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        print("\n" + "="*80)
        print(f"📝 Question: {question}")
        print("="*80)
        
        # Caution check: empty or whitespace question
        if not question or not str(question).strip():
            msg = "⚠️ Caution: Empty question provided. Please provide a non-empty question."
            print(f"\n{msg}")
            return {
                'question': question,
                'answer': msg,
                'sources': [],
                'context_chunks': 0,
                'response_time': 0.0
            }
        
        context = self.retrieve_context(question)
        
        if not context.get('success'):
            return {
                'question': question,
                'answer': f"Error: {context.get('message', 'Unknown error')}",
                'sources': [],
                'context_chunks': 0,
                'response_time': time.time() - start_time
            }
        
        if context.get('low_confidence'):
            print("\n⚠️ Caution: Low similarity confidence (< 25%). Context may not directly answer this question.")
        
        prompt = self.build_prompt(context)
        response = self.generate_response(prompt)
        
        elapsed_time = time.time() - start_time
        
        if response.get('success'):
            answer = response['answer']
            
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat(),
                'sources': context.get('sources', [])
            })
            
            print(f"\n✅ Response ({elapsed_time:.2f}s, {response.get('tokens', 0)} tokens):")
            print("-"*80)
            print(answer)
            print("-"*80)
            
            if context.get('sources'):
                print(f"\n📚 Sources: {', '.join(context['sources'])}")
            
            if self.verbose:
                print(f"\n📊 Retrieved {len(context.get('chunks', []))} chunks with scores:")
                for i, (chunk, score, meta) in enumerate(zip(
                    context.get('chunks', [])[:10],
                    context.get('scores', [])[:10],
                    context.get('metadatas', [])[:10]
                )):
                    print(f"   [{i+1}] {score:.2%} - {meta.get('source_file', 'Unknown')[:40]}")
            
            return {
                'question': question,
                'answer': answer,
                'sources': context.get('sources', []),
                'context_chunks': len(context.get('chunks', [])),
                'relevance_scores': context.get('scores', []),
                'response_time': elapsed_time,
                'tokens_generated': response.get('tokens', 0),
                'conversation_turns': len(self.conversation_history)
            }
        else:
            return {
                'question': question,
                'answer': f"Error: {response.get('error', 'Unknown error')}",
                'sources': [],
                'context_chunks': 0,
                'response_time': elapsed_time
            }
    
    def interactive_mode(self):
        """Run interactive conversation mode"""
        print("\n" + "="*80)
        print("🎯 FACE - RAG Interface (Frontend for Hierarchical Knowledge Base)")
        print("="*80)
        print(f"📚 ChromaDB: {self.processor.chunks_collection.count()} chunks total")
        print(f"🤖 Ollama Model: {self.ollama_model}")
        print(f"📊 Context: {self.max_context_chunks} chunks, Summaries: {self.include_summaries}")
        print("\nCommands:")
        print("  /clear    - Clear conversation history")
        print("  /stats    - Show database statistics")
        print("  /sources  - Show all documents in database")
        print("  /clusters - Show cluster summaries")
        print("  /help     - Show this help")
        print("  /quit     - Exit")
        print("="*80)
        
        while True:
            try:
                user_input = input("\n❓ You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == '/quit':
                    print("\n👋 Goodbye!")
                    break
                    
                elif user_input.lower() == '/clear':
                    self.conversation_history = []
                    print("✅ Conversation history cleared")
                    continue
                    
                elif user_input.lower() == '/stats':
                    self.show_stats()
                    continue
                    
                elif user_input.lower() == '/sources':
                    self.show_sources()
                    continue
                    
                elif user_input.lower() == '/clusters':
                    self.show_clusters()
                    continue
                    
                elif user_input.lower() == '/help':
                    self.show_help()
                    continue
                
                result = self.query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def show_stats(self):
        """Show database statistics from local ChromaDB"""
        try:
            total_chunks = self.processor.chunks_collection.count()
            total_summaries = self.processor.summaries_collection.count()
            
            print("\n📊 DATABASE STATISTICS (ChromaDB)")
            print("="*60)
            print(f"Total chunks: {total_chunks}")
            print(f"Total summaries: {total_summaries}")
            print(f"Session queries: {len(self.conversation_history)}")
            
            if total_chunks > 0:
                result = self.processor.chunks_collection.get(include=["metadatas"], limit=10000)
                sources = {}
                for meta in result['metadatas']:
                    source = meta.get('source_file', 'Unknown')
                    sources[source] = sources.get(source, 0) + 1
                
                print(f"\nDocument sources:")
                for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                    print(f"  • {source}: {count} chunks")
                
        except Exception as e:
            print(f"Error getting stats: {e}")
    
    def show_sources(self):
        """Show all documents in the database"""
        try:
            result = self.processor.chunks_collection.get(include=["metadatas"], limit=10000)
            sources = {}
            for meta in result['metadatas']:
                source = meta.get('source_file', 'Unknown')
                sources[source] = sources.get(source, 0) + 1
            
            print("\n📚 DOCUMENTS IN DATABASE (ChromaDB)")
            print("="*60)
            for source, count in sorted(sources.items()):
                print(f"  • {source}: {count} chunks")
                
        except Exception as e:
            print(f"Error getting sources: {e}")
    
    def show_clusters(self):
        """Show cluster summaries"""
        try:
            df = self.processor.get_summary_tree()
            if df.empty:
                print("\n📊 No cluster summaries found")
                return
            
            print("\n📊 CLUSTER SUMMARIES")
            print("="*60)   
            
            for level in sorted(df['level'].unique()):
                level_df = df[df['level'] == level]
                print(f"\nLevel {level} ({len(level_df)} clusters):")
                print("-"*40)
                
                for _, row in level_df.iterrows():
                    print(f"\n  Cluster {row['cluster_id']}:")
                    print(f"    Chunks: {row['chunks_count']}")
                    print(f"    Silhouette: {row['silhouette_score']:.3f}")
                    print(f"    Preview: {row['summary_preview'][:100]}...")
                    
        except Exception as e:
            print(f"Error getting clusters: {e}")
    
    def show_help(self):
        """Show help information"""
        print("\n📖 HELP")
        print("="*60)
        print("Commands:")
        print("  /clear    - Clear conversation history")
        print("  /stats    - Show database statistics")
        print("  /sources  - Show all documents in database")
        print("  /clusters - Show cluster summaries")
        print("  /help     - Show this help")
        print("  /quit     - Exit")
        print("\nTips:")
        print("  • Ask questions about your documents")
        print("  • The system retrieves relevant chunks using HNSW search")
        print("  • Responses include source attribution")
        print("  • Conversation history provides context for follow-up questions")
    
    def batch_query(self, questions: List[str], output_file: str = None) -> List[Dict]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of questions
            output_file: Optional file to save results
            
        Returns:
            List of results
        """
        results = []
        for i, question in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] Processing...")
            result = self.query(question)
            results.append(result)
            time.sleep(0.5)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Results saved to {output_file}")
        
        return results
    
    def export_conversation(self, filename: str = None):
        """Export conversation history to file"""
        if not self.conversation_history:
            print("No conversation history to export")
            return
        
        if filename is None:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'session': self.current_session,
            'model': self.ollama_model,
            'total_queries': len(self.conversation_history),
            'conversation': self.conversation_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Conversation exported to {filename}")


# ========== MAIN EXECUTION ==========

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FACE - RAG Interface for Hierarchical Knowledge Base (Local Version)')
    parser.add_argument('--model', default='gemma3:1b', help='Ollama model name')
    parser.add_argument('--embedding-model', default='nomic-embed-text', help='Ollama embedding model')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Ollama API URL')
    parser.add_argument('--chroma-path', default='./hierarchical_chroma_db', help='Local ChromaDB directory path')
    parser.add_argument('--chroma-host', default=None, help='ChromaDB host (optional, for remote server)')
    parser.add_argument('--chroma-port', type=int, default=None, help='ChromaDB port (optional, for remote server)')
    parser.add_argument('--chunks', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument('--no-summaries', action='store_true', help='Disable cluster summaries')
    parser.add_argument('--temperature', type=float, default=0.3, help='LLM temperature')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--query', type=str, help='Single query (non-interactive mode)')
    parser.add_argument('--batch', type=str, help='File with list of questions (one per line)')
    parser.add_argument('--export', type=str, help='Export conversation to file')
    
    args = parser.parse_args()
    
    # Initialize interface with local services
    interface = RAGInterface(
        ollama_model=args.model,
        ollama_embedding_model=args.embedding_model,
        ollama_url=args.ollama_url,
        chroma_path=args.chroma_path,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        max_context_chunks=args.chunks,
        include_summaries=not args.no_summaries,
        temperature=args.temperature,
        verbose=not args.quiet
    )
    
    # Check if database has content
    total_chunks = interface.processor.chunks_collection.count()
    if total_chunks == 0:
        print("\n⚠️ Database is empty!")
        print("Please add documents first using main.py")
        print("Example: python main.py")
        return
    
    # Handle different modes
    if args.query:
        result = interface.query(args.query)
        print(f"\n{result['answer']}")
        
    elif args.batch:
        if not os.path.exists(args.batch):
            print(f"\n❌ Batch file not found: {args.batch}")
            return
            
        with open(args.batch, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(questions)} questions...")
        output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results = interface.batch_query(questions, output_file)
        
    else:
        interface.interactive_mode()
        
        if args.export:
            interface.export_conversation(args.export)


if __name__ == "__main__":
    main()