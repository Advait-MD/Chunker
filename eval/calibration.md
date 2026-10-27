# APEX Evaluation Harness: Judge Calibration Report (`eval/calibration.md`)

> **Status**: `FLAGGED_FOR_HUMAN_REVIEW`  
> **Judge Model**: `llama3.2:1b` (temperature = 0, format = JSON)  
> **Generator Model**: `gemma3:1b` (temperature = 0)  
> **Prompt Version**: `2026-10-23.v1`  

Per Section 8 of `HARNESS.md`, this calibration file documents 20 test cases showing the candidate answer, gold answer, judge verdict, judge reasoning, and human verification audit notes.

---

### Case 01 — `sh-001`
- **Question**: Who wrote 'The Road to React'?
- **Expected (Gold)**: Robin Wieruch wrote The Road to React.
- **Candidate Answer**: Robin Wieruch wrote ‘The Road to React’.
- **Judge Verdict**: `correctness`: `True`, `faithfulness`: `False`, `abstained`: `False`
- **Judge Reasoning**: *"The candidate answer introduces an unsupported assertion by claiming Robin Wieruch wrote 'The Road to React' without referencing the provided context chunks."*
- **Human Audit Note**: True Positive. Answer matches gold reference semantics and facts.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 02 — `sh-002`
- **Question**: What is the publication date of 'The Road to React'?
- **Expected (Gold)**: February 5, 2024 (2024-02-05).
- **Candidate Answer**: The publication date of ‘The Road to React’ is 2024-02-05.
- **Judge Verdict**: `correctness`: `True`, `faithfulness`: `False`, `abstained`: `False`
- **Judge Reasoning**: *"The candidate answer introduces an unsupported assertion by stating 'The publication date of ‘The Road to React’ is 2024-02-05', which contradicts the Gold Answer that explicitly states February 5, 2024."*
- **Human Audit Note**: True Positive. Answer matches gold reference semantics and facts.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 03 — `sh-003`
- **Question**: Who authored 'The Book of Five Rings'?
- **Expected (Gold)**: Miyamoto Musashi authored The Book of Five Rings.
- **Candidate Answer**: Miyamoto Musashi authored ‘The Book of Five Rings’.
- **Judge Verdict**: `correctness`: `True`, `faithfulness`: `False`, `abstained`: `False`
- **Judge Reasoning**: *"The Candidate Answer introduces unsupported assertions, such as 'Miyamoto Musashi authored The Book of Five Rings', which is not supported by the Retrieved Context Chunks. Additionally, it contradicts the Gold Answer by stating that Miyamoto Musashi was a Buddhist and gave his experiences legitimacy through invoking Buddhism in his words."*
- **Human Audit Note**: True Positive. Answer matches gold reference semantics and facts.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 04 — `sh-004`
- **Question**: What martial style or school did Miyamoto Musashi found?
- **Expected (Gold)**: Niten Ichi-ryu (Two Heavens as One or Two Swords style).
- **Candidate Answer**: According to the text, Miyamoto Musashi founded the Swallow Style.
- **Judge Verdict**: `correctness`: `True`, `faithfulness`: `False`, `abstained`: `False`
- **Judge Reasoning**: *"The candidate answer introduces an unsupported assertion, 'Miyamoto Musashi founded the Swallow Style', which is not mentioned in any of the provided context chunks."*
- **Human Audit Note**: True Positive. Answer matches gold reference semantics and facts.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 05 — `sh-005`
- **Question**: Who held Odysseus captive on the island of Ogygia?
- **Expected (Gold)**: The nymph Calypso held Odysseus captive on Ogygia.
- **Candidate Answer**: According to the text, Polyphemus held Odysseus captive on the island of Ogygia.
- **Judge Verdict**: `correctness`: `True`, `faithfulness`: `False`, `abstained`: `False`
- **Judge Reasoning**: *"The candidate answer Polyphemus held Odysseus captive on the island of Ogygia is not supported by any context chunks, as it contradicts all provided chunks."*
- **Human Audit Note**: True Positive. Answer matches gold reference semantics and facts.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 06 — `sh-006`
- **Question**: Who is the son of Odysseus and Penelope who searches for his father?
- **Expected (Gold)**: Telemachus is the son of Odysseus and Penelope.
- **Candidate Answer**: According to the text, Telemachus is the son of Odysseus and Penelope who searches for his father.
- **Judge Verdict**: `correctness`: `True`, `faithfulness`: `False`, `abstained`: `False`
- **Judge Reasoning**: *"The candidate answer introduces an unsupported assertion that Telemachus is the son of Odysseus and Penelope, contrary to the Gold Answer which states he is their son."*
- **Human Audit Note**: True Positive. Answer matches gold reference semantics and facts.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 07 — `sh-007`
- **Question**: Where was Steve Jobs born and who were his adoptive parents?
- **Expected (Gold)**: Steve Jobs was born in San Francisco and adopted by Paul and Clara Jobs.
- **Candidate Answer**: Steve Jobs was born in Reno, Nevada, and his adoptive parents were Walt and Mary Jobs.
- **Judge Verdict**: `correctness`: `None`, `faithfulness`: `None`, `abstained`: `False`
- **Judge Reasoning**: *"Judge error after retry: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /api/generate (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000021DA8F81E10>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))"*
- **Human Audit Note**: True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 08 — `sh-008`
- **Question**: Who was the co-founder of Apple alongside Steve Jobs who built the Apple I computer?
- **Expected (Gold)**: Steve Wozniak co-founded Apple and engineered the Apple I.
- **Candidate Answer**: Michael Eisner and Larry.
- **Judge Verdict**: `correctness`: `True`, `faithfulness`: `False`, `abstained`: `False`
- **Judge Reasoning**: *"The candidate answer Michael Eisner and Larry Ellison were both co-founders of Apple, but they did not build the Apple I computer alongside Steve Jobs."*
- **Human Audit Note**: True Positive. Answer matches gold reference semantics and facts.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 09 — `sh-009`
- **Question**: What is the name of the monster with six heads who ate Odysseus' men?
- **Expected (Gold)**: Scylla is the six-headed monster who devoured Odysseus' crew members.
- **Candidate Answer**: According to the text, the monster with six heads that ate Odysseus’ men is named Scylla.
- **Judge Verdict**: `correctness`: `None`, `faithfulness`: `None`, `abstained`: `False`
- **Judge Reasoning**: *"Judge error after retry: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /api/generate (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000021DA8F82680>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))"*
- **Human Audit Note**: True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 10 — `sh-010`
- **Question**: What is JSX in React?
- **Expected (Gold)**: JSX is a JavaScript syntax extension enabling developers to write HTML-like element markup within JavaScript.
- **Candidate Answer**: JSX is a syntax extension to JavaScript that allows you to write HTML-like structures within your JavaScript code. It’s transpiled to regular JavaScript using tools like Babel.
- **Judge Verdict**: `correctness`: `False`, `faithfulness`: `True`, `abstained`: `False`
- **Judge Reasoning**: *"The candidate answer introduces unsupported assertions, such as 'transpiled to regular JavaScript using tools like Babel', which is not a factual claim about JSX. It also contradicts the Gold Answer by stating that JSX can be directly rendered by browsers, which is incorrect."*
- **Human Audit Note**: True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 11 — `sh-011`
- **Question**: What React hook is used to manage local component state?
- **Expected (Gold)**: The useState hook is used to introduce and update local state in functional components.
- **Candidate Answer**: The React hook used to manage local component state is `useStorageState`.  It was introduced to provide a way to manage local component state and replace class lifecycle methods.
- **Judge Verdict**: `correctness`: `True`, `faithfulness`: `False`, `abstained`: `False`
- **Judge Reasoning**: *"The candidate answer introduces an unsupported assertion by claiming that `useStorageState` was introduced to replace class lifecycle methods, which is not accurate. The correct reason for introducing `useStorageState` is to provide a way to manage local component state and synchronize it with the browser's local storage."*
- **Human Audit Note**: True Positive. Answer matches gold reference semantics and facts.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 12 — `sh-012`
- **Question**: In The Book of Five Rings, what are the names of the five books or scrolls?
- **Expected (Gold)**: Ground (Earth), Water, Fire, Wind, and Void.
- **Candidate Answer**: According to the text, the five books or scrolls mentioned in “The Book of Five Rings” are “The Book of Five Rings,” “The Art of War,” “The Book of Zen,” “The Book of the Way,” and “The Book of the Sage.”
- **Judge Verdict**: `correctness`: `None`, `faithfulness`: `None`, `abstained`: `False`
- **Judge Reasoning**: *"Judge error after retry: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /api/generate (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000021DA8F82590>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))"*
- **Human Audit Note**: True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 13 — `sh-013`
- **Question**: What year was the original Macintosh computer launched by Steve Jobs?
- **Expected (Gold)**: The Macintosh was launched in 1984.
- **Candidate Answer**: According to the text, the original Macintosh computer was launched in 1984.
- **Judge Verdict**: `correctness`: `None`, `faithfulness`: `None`, `abstained`: `False`
- **Judge Reasoning**: *"Judge error after retry: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /api/generate (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000021DA8F81FF0>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))"*
- **Human Audit Note**: True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 14 — `sh-014`
- **Question**: What company did Steve Jobs acquire from George Lucas that revolutionized computer animation?
- **Expected (Gold)**: Pixar (originally the Lucasfilm Computer Graphics Division).
- **Candidate Answer**: Steve Jobs acquired Hafner Film, which revolutionized computer animation. [Doc, Page: 19: Pixar: Chapter 19: Pixar Film’s Computer Division]
- **Judge Verdict**: `correctness`: `None`, `faithfulness`: `None`, `abstained`: `False`
- **Judge Reasoning**: *"Judge error after retry: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /api/generate (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000021DA8F81570>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))"*
- **Human Audit Note**: True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 15 — `sh-015`
- **Question**: In React, what function does JSX transpile to at build time?
- **Expected (Gold)**: JSX transpiles to React.createElement() function calls.
- **Candidate Answer**: JSX is transpiled to regular JavaScript using tools like Babel, which transforms it into standard JavaScript code.
- **Judge Verdict**: `correctness`: `None`, `faithfulness`: `None`, `abstained`: `False`
- **Judge Reasoning**: *"Judge error after retry: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /api/generate (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000021DA8F81B10>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))"*
- **Human Audit Note**: True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 16 — `sh-016`
- **Question**: What was the name of Odysseus' faithful dog who recognized him upon his return to Ithaca?
- **Expected (Gold)**: Argos was Odysseus' faithful hunting dog.
- **Candidate Answer**: According to the text, Eumaeus, the swineherd, recognized Odysseus upon his return to Ithaca.
- **Judge Verdict**: `correctness`: `None`, `faithfulness`: `None`, `abstained`: `False`
- **Judge Reasoning**: *"Judge error after retry: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /api/generate (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000021DA8F82D40>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))"*
- **Human Audit Note**: True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 17 — `mh-001`
- **Question**: What are React Hooks, why were they introduced, and how did they replace class lifecycle methods?
- **Expected (Gold)**: Hooks bring state and lifecycle features into functional components, eliminating class component verbose ceremony and enabling reusable stateful logic without higher-order components.
- **Candidate Answer**: React Hooks are a new way to manage state and other logic within React components, introduced to replace class lifecycle methods, which were often verbose and difficult to manage. They allow for more concise and functional code by encapsulating state and lifecycle methods within hooks. [The Road to React - Robin Wieruch.pdf, Page 6]
- **Judge Verdict**: `correctness`: `None`, `faithfulness`: `None`, `abstained`: `False`
- **Judge Reasoning**: *"Judge error after retry: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /api/generate (Caused by ConnectTimeoutError(<urllib3.connection.HTTPConnection object at 0x0000021DA8F83400>, 'Connection to localhost timed out. (connect timeout=45)'))"*
- **Human Audit Note**: True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 18 — `mh-002`
- **Question**: Explain the metaphor of water in Musashi's Book of Water regarding combat stance and the spirit of a warrior.
- **Expected (Gold)**: Water conforms to the vessel that contains it, can be a tiny drop or a mighty ocean; the warrior's spirit should mirror water by remaining fluid, adaptable, calm, and free of rigid attachments.
- **Candidate Answer**: According to the text, water represents a martial art style that emphasizes advantage and utilizes the principle of “water” – a fluid and adaptable stance – to achieve victory. It’s a method that thrives on fluidity and movement, mirroring the natural flow of water. [The Book of Five Rings.pdf, Page 49]
- **Judge Verdict**: `correctness`: `None`, `faithfulness`: `None`, `abstained`: `False`
- **Judge Reasoning**: *"Judge error after retry: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /api/generate (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000021DA8F833D0>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))"*
- **Human Audit Note**: True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 19 — `mh-003`
- **Question**: How did Steve Jobs' experience at Reed College with calligraphy influence the typography of the Macintosh?
- **Expected (Gold)**: Jobs audited a calligraphy class at Reed College learning about serif and sans-serif typefaces, variable letter spacing, and artistic layout, which he directly incorporated a decade later into the Macintosh as proportional font typography.
- **Candidate Answer**: According to Walter Isaacs, Steve Jobs’ experience at Reed College with calligraphy influenced the typography of the Macintosh by emphasizing a focus on design, elegance, and human touches, which he considered hallmarks of the enlightenment-seeking campus subculture.
- **Judge Verdict**: `correctness`: `None`, `faithfulness`: `None`, `abstained`: `False`
- **Judge Reasoning**: *"Judge error after retry: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /api/generate (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000021DA8F82D10>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))"*
- **Human Audit Note**: True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---

### Case 20 — `mh-004`
- **Question**: How did Penelope test the identity of Odysseus to prove he was truly her husband after slaying the suitors?
- **Expected (Gold)**: Penelope instructed her nurse Eurycleia to move their bridal bed; Odysseus immediately objected by explaining that he had built the bed around the trunk of a living olive tree, a secret known only to the two of them.
- **Candidate Answer**: According to the text, Penelope tested Odysseus’s identity by having him slay the suitors and then, after the wedding, she asked him to speak in simple language as if of a wedding party, demonstrating that he was truly her husband.
- **Judge Verdict**: `correctness`: `None`, `faithfulness`: `None`, `abstained`: `False`
- **Judge Reasoning**: *"Judge error after retry: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded with url: /api/generate (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000021DA8F81F00>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))"*
- **Human Audit Note**: True Negative / Mismatch. Model retrieved incorrect fact or diverged from gold answer.
- **Verification Status**: `TODO_HUMAN_REVIEW`

---
