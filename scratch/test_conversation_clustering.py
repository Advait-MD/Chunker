"""
Verification test for conversational memory storage, hierarchical clustering, and recall.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from main import HierarchicalClusterSummarizer
from face import RAGInterface
from modules.config import config
from modules.logger import setup_logger, get_logger

setup_logger(level="INFO")
logger = get_logger("test_conv_clustering")

def run_tests():
    print("\n" + "=" * 70)
    print("🧪 TEST: Conversational Memory Storage, Clustering, and RAG Recall")
    print("=" * 70)

    # 1. Initialize Processor
    print("\n[Step 1] Initializing HierarchicalClusterSummarizer...")
    processor = HierarchicalClusterSummarizer()
    
    initial_turns = processor.count_conversation_chunks()
    print(f"Initial conversation turns in memory: {initial_turns}")

    # 2. Add distinct conversation turns
    print("\n[Step 2] Storing distinct conversation turns across two distinct topic clusters...")
    test_chat_drone = "test_chat_drone_001"
    test_chat_web = "test_chat_web_002"

    drone_turns = [
        ("I am designing an autonomous quadcopter drone using PX4 autopilot and ROS 2 for orchard monitoring.",
         "That sounds like an impressive robotics project! PX4 and ROS 2 provide robust real-time flight controls and telemetry for agricultural robotics."),
        ("We decided to use solid-state LiDAR instead of stereo cameras for obstacle avoidance in dense orchard foliage.",
         "Solid-state LiDAR is superior in dense foliage because variable sunlight and shadowing often degrade optical stereo camera depth perception."),
        ("The drone payload includes an NVIDIA Jetson Orin Nano for on-board edge computing and YOLOv8 object detection.",
         "The Jetson Orin Nano provides great FP16 compute efficiency for running edge YOLOv8 detection under 15W power constraints.")
    ]

    web_turns = [
        ("For our web applications, our engineering team exclusively prefers React 18 with TypeScript and Tailwind CSS.",
         "React 18 with TypeScript and Tailwind CSS offers excellent type-safety, rapid styling workflows, and concurrent rendering capabilities."),
        ("We decided to standardize on Zustand for global client-side state management instead of Redux Toolkit because it avoids boilerplate.",
         "Zustand provides a minimalist hook-based store that drastically reduces boilerplate compared to Redux Toolkit while maintaining strict state immutability.")
    ]

    for i, (q, a) in enumerate(drone_turns):
        t_id = processor.add_conversation_turn(
            chat_id=test_chat_drone,
            user_text=q,
            assistant_text=a,
            turn_index=i,
            chat_title="Autonomous Drone Engineering"
        )
        print(f"  ✓ Stored Drone turn {i}: {t_id}")

    for i, (q, a) in enumerate(web_turns):
        t_id = processor.add_conversation_turn(
            chat_id=test_chat_web,
            user_text=q,
            assistant_text=a,
            turn_index=i,
            chat_title="Frontend Architecture"
        )
        print(f"  ✓ Stored Web turn {i}: {t_id}")

    current_turns = processor.count_conversation_chunks()
    print(f"\nTotal conversation turns in memory: {current_turns}")
    assert current_turns >= 5, f"Expected at least 5 conversation turns, found {current_turns}"

    # 3. Perform hierarchical clustering over conversation history
    print("\n[Step 3] Running hierarchical clustering over conversation history...")
    cluster_res = processor.cluster_conversations(max_levels=1, skip_bad_clusters=False)
    print(f"Clustering output: success={cluster_res.get('success')}, message={cluster_res.get('message')}")
    
    summary_tree = processor.get_conversation_summary_tree()
    print(f"\nConversation Summary Tree:\n{summary_tree[['summary_id', 'level', 'cluster_id', 'chunks_count', 'silhouette_score']] if not summary_tree.empty else 'Empty'}")
    assert processor.count_conversation_summaries() > 0, "Expected at least 1 conversation cluster summary"

    # 4. Test RAG Recall via RAGInterface
    print("\n[Step 4] Testing RAG Recall with conversational queries...")
    rag = RAGInterface(
        ollama_model=config.ollama.generation_model,
        ollama_embedding_model=config.ollama.embedding_model,
        chroma_path=config.chroma.path,
        verbose=False
    )

    # Query 1: Drone obstacle avoidance
    q1 = "Which obstacle avoidance sensor did I choose for my orchard drone project and why?"
    print(f"\n--- Query 1: '{q1}' ---")
    ctx1 = rag.retrieve_context(q1)
    print(f"  Retrieved primary chunks: {len(ctx1.get('chunks', []))}")
    print(f"  Retrieved conversation memories: {len(ctx1.get('conversation_memories', []))}")
    print(f"  Retrieved conversation topic themes: {len(ctx1.get('conversation_cluster_summaries', []))}")

    assert len(ctx1.get('conversation_memories', [])) > 0, "Expected conversation memory to be recalled for query 1"
    top_mem1 = ctx1['conversation_memories'][0]['text']
    print(f"  Top Recalled Memory: {top_mem1[:150]}...")
    assert "LiDAR" in top_mem1 or "drone" in top_mem1.lower() or "px4" in top_mem1.lower()

    # Generate answer 1
    prompt1 = rag.build_prompt(ctx1)
    ans1_res = rag.generate_response(prompt1)
    print(f"\n🤖 Gemma 3 Answer 1:\n{ans1_res.get('answer')}\n")
    assert "lidar" in ans1_res.get('answer', '').lower()

    # Query 2: Frontend state management
    q2 = "What library did our team standardize on for global client-side state management?"
    print(f"\n--- Query 2: '{q2}' ---")
    ctx2 = rag.retrieve_context(q2)
    print(f"  Retrieved primary chunks: {len(ctx2.get('chunks', []))}")
    print(f"  Retrieved conversation memories: {len(ctx2.get('conversation_memories', []))}")
    print(f"  Retrieved conversation topic themes: {len(ctx2.get('conversation_cluster_summaries', []))}")

    assert len(ctx2.get('conversation_memories', [])) > 0, "Expected conversation memory to be recalled for query 2"
    top_mem2 = ctx2['conversation_memories'][0]['text']
    print(f"  Top Recalled Memory: {top_mem2[:150]}...")
    assert "Zustand" in top_mem2 or "state management" in top_mem2.lower()

    # Generate answer 2
    prompt2 = rag.build_prompt(ctx2)
    ans2_res = rag.generate_response(prompt2)
    print(f"\n🤖 Gemma 3 Answer 2:\n{ans2_res.get('answer')}\n")
    assert "zustand" in ans2_res.get('answer', '').lower()

    print("\n" + "=" * 70)
    print("✅ ALL CONVERSATIONAL MEMORY & CLUSTERING TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
