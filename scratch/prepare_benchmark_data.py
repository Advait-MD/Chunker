"""
scratch/prepare_benchmark_data.py — Prepares and caches standard benchmark datasets:
1. Cinderella fairytale (for qualitative multi-hop comparison, Tables 12-13)
2. QuALITY long-form stories and MC questions (Tables 2, 4, 7, 8, 9)
3. NarrativeQA stories and question-answer pairs (Table 1)
4. QASPER research papers and question-answer pairs (Table 3)
"""

import os
import sys
import json
import urllib.request
import tarfile
import io

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark_data")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Cinderella Fairytale (Tables 12 & 13)
cinderella_file = os.path.join(DATA_DIR, "cinderella.json")
if not os.path.exists(cinderella_file):
    print("Preparing Cinderella benchmark dataset...")
    with open("cinderella_sample.txt", "r", encoding="utf-8") as f:
        cinderella_text = f.read()

    cinderella_data = {
        "title": "Cinderella",
        "text": cinderella_text,
        "questions": [
            {
                "id": "cind-01",
                "question": "What is the central theme of the story?",
                "type": "thematic_multi_hop",
                "paper_reference": "Table 12 / 13"
            },
            {
                "id": "cind-02",
                "question": "How did Cinderella find a happy ending?",
                "type": "narrative_synthesis",
                "paper_reference": "Table 12 / 13"
            }
        ]
    }
    with open(cinderella_file, "w", encoding="utf-8") as f:
        json.dump(cinderella_data, f, indent=2)
    print(f"✓ Saved {cinderella_file}")

# 2. QuALITY benchmark dataset (Stories 1, 2 and questions)
quality_file = os.path.join(DATA_DIR, "quality_stories.json")
if not os.path.exists(quality_file):
    print("Fetching QuALITY stories from NYU-MLL repository...")
    url = "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    stories = []
    with urllib.request.urlopen(req) as resp:
        # Read first 3 unique stories
        seen_articles = set()
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            art_id = item.get("article_id")
            if art_id not in seen_articles:
                seen_articles.add(art_id)
                stories.append({
                    "story_id": f"Story {len(stories) + 1}",
                    "article_id": art_id,
                    "title": item.get("title", ""),
                    "author": item.get("author", ""),
                    "article": item.get("article", ""),
                    "questions": [
                        {
                            "question_id": q.get("question_unique_id", f"q{i}"),
                            "question": q.get("question", ""),
                            "options": q.get("options", []),
                            "gold_label": q.get("gold_label", 0),
                            "difficult": bool(q.get("difficult", 0)),
                        }
                        for i, q in enumerate(item.get("questions", []))
                    ]
                })
            if len(stories) >= 3:
                break

    with open(quality_file, "w", encoding="utf-8") as f:
        json.dump(stories, f, indent=2)
    print(f"✓ Saved {len(stories)} QuALITY stories to {quality_file}")

# 3. NarrativeQA benchmark dataset (Table 1)
narrativeqa_file = os.path.join(DATA_DIR, "narrativeqa_stories.json")
if not os.path.exists(narrativeqa_file):
    print("Fetching NarrativeQA samples via datasets library...")
    try:
        from datasets import load_dataset
        ds = load_dataset("narrativeqa", split="test", streaming=True)
        narrative_items = []
        for item in ds:
            doc = item.get("document", {})
            summary_text = doc.get("summary", {}).get("text", "")
            q_text = item.get("question", {}).get("text", "")
            answers = [a["text"] for a in item.get("answers", [])]
            if summary_text and q_text and answers:
                narrative_items.append({
                    "id": item.get("document", {}).get("id", f"nqa_{len(narrative_items)}"),
                    "title": doc.get("kind", "NarrativeQA Story"),
                    "text": summary_text,
                    "question": q_text,
                    "answers": answers
                })
            if len(narrative_items) >= 8:
                break

        with open(narrativeqa_file, "w", encoding="utf-8") as f:
            json.dump(narrative_items, f, indent=2)
        print(f"✓ Saved {len(narrative_items)} NarrativeQA items to {narrativeqa_file}")
    except Exception as e:
        print(f"Warning: Failed to load NarrativeQA ({e})")

# 4. QASPER benchmark dataset (Table 3)
qasper_file = os.path.join(DATA_DIR, "qasper_papers.json")
if not os.path.exists(qasper_file):
    print("Fetching QASPER papers from AllenAI S3...")
    try:
        url = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        papers = []
        with urllib.request.urlopen(req) as resp:
            tf = tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz")
            f = tf.extractfile("qasper-dev-v0.3.json")
            data = json.load(f)
            for k in list(data.keys())[:3]:
                p = data[k]
                # Combine full text from sections
                full_text_paras = [p.get("abstract", "")]
                for sec in p.get("full_text", []):
                    full_text_paras.extend(sec.get("paragraphs", []))
                full_text = "\n\n".join(full_text_paras)

                qa_list = []
                for q in p.get("qas", []):
                    q_text = q.get("question", "")
                    gold_answers = []
                    for ans_entry in q.get("answers", []):
                        ans_dict = ans_entry.get("answer", {})
                        if ans_dict.get("extractive_spans"):
                            gold_answers.extend(ans_dict["extractive_spans"])
                        elif ans_dict.get("free_form_answer"):
                            gold_answers.append(ans_dict["free_form_answer"])
                        elif ans_dict.get("yes_no") is not None:
                            gold_answers.append("Yes" if ans_dict["yes_no"] else "No")
                    if gold_answers:
                        qa_list.append({
                            "question": q_text,
                            "gold_answers": gold_answers
                        })

                if qa_list:
                    papers.append({
                        "paper_id": k,
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", ""),
                        "full_text": full_text,
                        "qas": qa_list
                    })

        with open(qasper_file, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2)
        print(f"✓ Saved {len(papers)} QASPER papers to {qasper_file}")
    except Exception as e:
        print(f"Warning: Failed to load QASPER ({e})")
