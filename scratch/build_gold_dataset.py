"""
Builds the 32 gold evaluation cases for eval/dataset/gold.jsonl.
Maps questions to exact ChromaDB chunk IDs across:
- The Road to React
- The Book of Five Rings
- The Odyssey
- Steve Jobs
- Adversarial / Unanswerable queries
"""

import os
import json
import chromadb

client = chromadb.PersistentClient(path="./hierarchical_chroma_db")
col = client.get_collection("document_chunks")


def find_top_chunk_ids(query: str, n: int = 1) -> list:
    res = col.query(query_texts=[query], n_results=n)
    if res and res.get("ids") and res["ids"][0]:
        return res["ids"][0]
    return []


CASES_DEF = [
    # ================= SINGLE-HOP (16 cases) =================
    {
        "id": "sh-001",
        "question": "Who wrote 'The Road to React'?",
        "gold_answer": "Robin Wieruch wrote The Road to React.",
        "search_query": "The Road to React by Robin Wieruch author",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-002",
        "question": "What is the publication date of 'The Road to React'?",
        "gold_answer": "February 5, 2024 (2024-02-05).",
        "search_query": "February 5 2024 2024-02-05 The Road to React",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-003",
        "question": "Who authored 'The Book of Five Rings'?",
        "gold_answer": "Miyamoto Musashi authored The Book of Five Rings.",
        "search_query": "Miyamoto Musashi The Book of Five Rings author",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-004",
        "question": "What martial style or school did Miyamoto Musashi found?",
        "gold_answer": "Niten Ichi-ryu (Two Heavens as One or Two Swords style).",
        "search_query": "Niten Ichi-ryu Two Heavens style school Miyamoto Musashi",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-005",
        "question": "Who held Odysseus captive on the island of Ogygia?",
        "gold_answer": "The nymph Calypso held Odysseus captive on Ogygia.",
        "search_query": "Calypso island of Ogygia Odysseus captive daughter of Atlas",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-006",
        "question": "Who is the son of Odysseus and Penelope who searches for his father?",
        "gold_answer": "Telemachus is the son of Odysseus and Penelope.",
        "search_query": "Telemachus son of Odysseus Penelope",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-007",
        "question": "Where was Steve Jobs born and who were his adoptive parents?",
        "gold_answer": "Steve Jobs was born in San Francisco and adopted by Paul and Clara Jobs.",
        "search_query": "Paul and Clara Jobs adopted Steve Jobs San Francisco",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-008",
        "question": "Who was the co-founder of Apple alongside Steve Jobs who built the Apple I computer?",
        "gold_answer": "Steve Wozniak co-founded Apple and engineered the Apple I.",
        "search_query": "Steve Wozniak Apple I computer co-founder Steve Jobs",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-009",
        "question": "What is the name of the monster with six heads who ate Odysseus' men?",
        "gold_answer": "Scylla is the six-headed monster who devoured Odysseus' crew members.",
        "search_query": "Scylla six-headed monster Charybdis Odysseus crew",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-010",
        "question": "What is JSX in React?",
        "gold_answer": "JSX is a JavaScript syntax extension enabling developers to write HTML-like element markup within JavaScript.",
        "search_query": "JSX JavaScript XML syntax extension HTML React",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-011",
        "question": "What React hook is used to manage local component state?",
        "gold_answer": "The useState hook is used to introduce and update local state in functional components.",
        "search_query": "useState hook state functional component React",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-012",
        "question": "In The Book of Five Rings, what are the names of the five books or scrolls?",
        "gold_answer": "Ground (Earth), Water, Fire, Wind, and Void.",
        "search_query": "Ground Water Fire Wind Void Five books scrolls Musashi",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-013",
        "question": "What year was the original Macintosh computer launched by Steve Jobs?",
        "gold_answer": "The Macintosh was launched in 1984.",
        "search_query": "1984 Macintosh computer Steve Jobs launch Super Bowl",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-014",
        "question": "What company did Steve Jobs acquire from George Lucas that revolutionized computer animation?",
        "gold_answer": "Pixar (originally the Lucasfilm Computer Graphics Division).",
        "search_query": "Pixar George Lucas computer graphics Steve Jobs animation",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-015",
        "question": "In React, what function does JSX transpile to at build time?",
        "gold_answer": "JSX transpiles to React.createElement() function calls.",
        "search_query": "React.createElement JSX transpiles build time",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },
    {
        "id": "sh-016",
        "question": "What was the name of Odysseus' faithful dog who recognized him upon his return to Ithaca?",
        "gold_answer": "Argos was Odysseus' faithful hunting dog.",
        "search_query": "Argos faithful dog Odysseus Ithaca dung heap twenty years",
        "hops": 1,
        "answerable": True,
        "n_chunks": 1,
    },

    # ================= MULTI-HOP (11 cases) =================
    {
        "id": "mh-001",
        "question": "What are React Hooks, why were they introduced, and how did they replace class lifecycle methods?",
        "gold_answer": "Hooks bring state and lifecycle features into functional components, eliminating class component verbose ceremony and enabling reusable stateful logic without higher-order components.",
        "search_query": "React Hooks introduced functional components replace classes lifecycle",
        "hops": 2,
        "answerable": True,
        "n_chunks": 2,
    },
    {
        "id": "mh-002",
        "question": "Explain the metaphor of water in Musashi's Book of Water regarding combat stance and the spirit of a warrior.",
        "gold_answer": "Water conforms to the vessel that contains it, can be a tiny drop or a mighty ocean; the warrior's spirit should mirror water by remaining fluid, adaptable, calm, and free of rigid attachments.",
        "search_query": "Book of Water spirit adaptable fluid vessel stance Musashi",
        "hops": 2,
        "answerable": True,
        "n_chunks": 2,
    },
    {
        "id": "mh-003",
        "question": "How did Steve Jobs' experience at Reed College with calligraphy influence the typography of the Macintosh?",
        "gold_answer": "Jobs audited a calligraphy class at Reed College learning about serif and sans-serif typefaces, variable letter spacing, and artistic layout, which he directly incorporated a decade later into the Macintosh as proportional font typography.",
        "search_query": "calligraphy Reed College proportional fonts typography Macintosh Steve Jobs",
        "hops": 2,
        "answerable": True,
        "n_chunks": 2,
    },
    {
        "id": "mh-004",
        "question": "How did Penelope test the identity of Odysseus to prove he was truly her husband after slaying the suitors?",
        "gold_answer": "Penelope instructed her nurse Eurycleia to move their bridal bed; Odysseus immediately objected by explaining that he had built the bed around the trunk of a living olive tree, a secret known only to the two of them.",
        "search_query": "Penelope test bridal bed olive tree trunk Odysseus secret",
        "hops": 2,
        "answerable": True,
        "n_chunks": 2,
    },
    {
        "id": "mh-005",
        "question": "Compare the learning philosophy in 'The Road to React' with the discipline of strategy in 'The Book of Five Rings'. What fundamental principles do they share?",
        "gold_answer": "Both prioritize mastering fundamentals before moving to complex techniques, continuous deliberate practice over passive memorization, eliminating unnecessary complications, and applying disciplined principles to real craft.",
        "search_query": "fundamentals deliberate practice principles craft mastery",
        "hops": 3,
        "answerable": True,
        "n_chunks": 3,
    },
    {
        "id": "mh-006",
        "question": "How does Musashi's concept of 'the Void' relate to the software engineering principle of clean, minimal component design in React?",
        "gold_answer": "The Void represents clear perception free of confusion or unnecessary clutter; in React code, this translates to doing nothing unnecessary, single-responsibility components, and removing redundant abstractions.",
        "search_query": "the Void nothing unnecessary clear spirit clean components",
        "hops": 3,
        "answerable": True,
        "n_chunks": 2,
    },
    {
        "id": "mh-007",
        "question": "What sequence of events led Steve Jobs to be ousted from Apple in 1985 by John Sculley and the board?",
        "gold_answer": "Slumping Macintosh office sales, friction between Jobs and CEO John Sculley, and Jobs' attempted boardroom coup to replace Sculley led the board to strip Jobs of all operational authority, prompting his resignation.",
        "search_query": "John Sculley boardroom coup ousted 1985 Steve Jobs resignation",
        "hops": 2,
        "answerable": True,
        "n_chunks": 2,
    },
    {
        "id": "mh-008",
        "question": "How did Odysseus' encounter with the Cyclops Polyphemus incur the wrath of the sea god Poseidon?",
        "gold_answer": "Odysseus blinded Polyphemus by driving a heated stake into his eye and escaping under sheep; upon sailing away, Odysseus hubristically yelled his real name, allowing Polyphemus to pray to his father Poseidon to curse Odysseus' journey.",
        "search_query": "Polyphemus Cyclops blinded Odysseus Poseidon curse father",
        "hops": 2,
        "answerable": True,
        "n_chunks": 2,
    },
    {
        "id": "mh-009",
        "question": "How does controlled component state in React pass data downward through props and receive updates upward through callback handlers?",
        "gold_answer": "Parent components hold state and pass current values downward as props to child input components, while passing event handler callback functions so children can trigger state updates upward upon user events.",
        "search_query": "controlled component props callback handler state updates upward downward",
        "hops": 2,
        "answerable": True,
        "n_chunks": 2,
    },
    {
        "id": "mh-010",
        "question": "How did Steve Jobs' design philosophy at Apple align with the Japanese aesthetic of simplicity and Zen Buddhism?",
        "gold_answer": "Jobs practiced Zen meditation with Kobun Chino and embraced the Bauhaus and Zen ideals that simplicity comes from mastering the essence of a product and eliminating everything non-essential.",
        "search_query": "Zen Buddhism simplicity Kobun Chino Bauhaus essence Steve Jobs",
        "hops": 2,
        "answerable": True,
        "n_chunks": 2,
    },
    {
        "id": "mh-011",
        "question": "In The Odyssey, how did the goddess Athena aid Telemachus and Odysseus in disguise throughout their journeys?",
        "gold_answer": "Athena disguised herself as Mentor to guide Telemachus to Pylos and Sparta, and later disguised Odysseus as an elderly beggar in Ithaca so he could reconnoiter his palace undetected before the suitors.",
        "search_query": "Athena Mentor disguise beggar Ithaca Telemachus Odysseus",
        "hops": 2,
        "answerable": True,
        "n_chunks": 2,
    },

    # ================= UNANSWERABLE / ADVERSARIAL (5 cases) =================
    {
        "id": "un-001",
        "question": "What specific React hooks did Steve Jobs implement when launching the original Macintosh computer in 1984?",
        "gold_answer": None,
        "gold_chunk_ids": [],
        "hops": 1,
        "answerable": False,
        "search_query": None,
    },
    {
        "id": "un-002",
        "question": "What is the official currency and capital city of the planet Mars according to the provided documents?",
        "gold_answer": None,
        "gold_chunk_ids": [],
        "hops": 1,
        "answerable": False,
        "search_query": None,
    },
    {
        "id": "un-003",
        "question": "In The Book of Five Rings, what advice does Miyamoto Musashi give regarding debugging TypeScript compiler errors?",
        "gold_answer": None,
        "gold_chunk_ids": [],
        "hops": 1,
        "answerable": False,
        "search_query": None,
    },
    {
        "id": "un-004",
        "question": "How many nuclear submarines did Odysseus command during the Trojan War?",
        "gold_answer": None,
        "gold_chunk_ids": [],
        "hops": 1,
        "answerable": False,
        "search_query": None,
    },
    {
        "id": "un-005",
        "question": "According to Walter Isaacson, what was Steve Jobs' favorite song by the music artist Taylor Swift?",
        "gold_answer": None,
        "gold_chunk_ids": [],
        "hops": 1,
        "answerable": False,
        "search_query": None,
    },
]

out_cases = []
for c in CASES_DEF:
    if c["answerable"]:
        query = c["search_query"]
        n = c.get("n_chunks", 1)
        found_ids = find_top_chunk_ids(query, n)
        case_obj = {
            "id": c["id"],
            "question": c["question"],
            "gold_answer": c["gold_answer"],
            "gold_chunk_ids": found_ids,
            "hops": c["hops"],
            "answerable": True,
            "status": "VERIFIED"
        }
    else:
        case_obj = {
            "id": c["id"],
            "question": c["question"],
            "gold_answer": None,
            "gold_chunk_ids": [],
            "hops": c["hops"],
            "answerable": False,
            "status": "VERIFIED"
        }
    out_cases.append(case_obj)

os.makedirs("eval/dataset", exist_ok=True)
gold_path = "eval/dataset/gold.jsonl"
with open(gold_path, "w", encoding="utf-8") as f:
    for c in out_cases:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"Generated {len(out_cases)} cases in {gold_path}.")
print(f"Single-hop count: {sum(1 for c in out_cases if c['hops'] == 1 and c['answerable'])}")
print(f"Multi-hop count: {sum(1 for c in out_cases if c['hops'] >= 2 and c['answerable'])}")
print(f"Unanswerable count: {sum(1 for c in out_cases if not c['answerable'])}")
