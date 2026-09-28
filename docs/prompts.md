# Generalized Prompt Paradigms & Evaluation Framework

> **Document Purpose**: This document details the generalized, task-agnostic prompting taxonomy, verbatim system/user templates, and experimental operationalization used in *PromptEnergy-Bench* (for NAACL / ACL submission).

---

## 1. Generalized Prompting Paradigms

In modern LLM empirical research (e.g. Wei et al., Kojima et al., Brown et al., Wang et al.), prompting techniques are grouped by **Prompting Paradigm (Meta-Strategy)** rather than dataset-specific scripts. PromptEnergy-Bench operationalizes **6 distinct paradigms**:

```
                                  ┌─────────────────────────────────────────┐
                                  │      Prompting Strategy Paradigms       │
                                  └────────────────────┬────────────────────┘
                                                       │
             ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
             │                                         │                                         │
    ┌────────┴────────┐                       ┌────────┴────────┐                       ┌────────┴────────┐
    │ Direct Inference│                       │    Reasoning    │                       │Context-Grounded │
    ├─────────────────┤                       ├─────────────────┤                       ├─────────────────┤
    │1. Zero-Shot     │                       │3. Zero-Shot CoT │                       │6. Retrieval-    │
    │   Direct        │                       │4. Short CoT     │                       │   Augmented Gen │
    │2. Few-Shot (3)  │                       │5. Long CoT      │                       │   (RAG)         │
    └─────────────────┘                       └─────────────────┘                       └─────────────────┘
```

### Paradigm Comparison Table

| Paradigm / Strategy | In-Context Exemplars | Reasoning Directive | Prompt Structure | Energy Profile Hypothesis |
| :--- | :---: | :--- | :--- | :--- |
| **1. Zero-Shot Direct** | 0 | None (Direct answer only) | `Task: {input} \n Answer:` | **Minimal Prefill & Decode Energy** (Baseline) |
| **2. Few-Shot (3-Shot)** | 3 | None (Demonstrates format) | `Example 1... Target: {input}` | **High Prefill Energy**, Low Decode Energy |
| **3. Zero-Shot CoT** | 0 | Standard (*"Let's think step by step"*) | `{input} \n Let's think step by step.` | Low Prefill Energy, **Moderate Decode Energy** |
| **4. Short CoT** | 0 | Budget-constrained ($\le 2$ steps) | `{input} \n Brief 1-2 step reasoning:` | Low Prefill Energy, **Pareto-Optimal Decode Energy** |
| **5. Long CoT** | 0 | Comprehensive / Detailed | `{input} \n Detailed reasoning:` | Low Prefill Energy, **Maximal Decode Energy** |
| **6. RAG / Context** | 0 | Context-grounded inference | `Context: {context} \n {input}` | **Variable Prefill Energy**, Moderate Decode Energy |

---

## 2. Universal Publication-Ready Prompt Templates

All models receive standardized dual-turn chat messages (`system` and `user`).

### Paradigm 1: Zero-Shot Direct (Raw / Standard Instruction)
* **System Prompt**:
  ```text
  You are an expert AI assistant.
  Provide a direct, concise, and accurate answer to the user's question or task.
  Do not include unnecessary conversational filler, preambles, or explanations.
  ```
* **User Template**:
  ```text
  Task:
  {task_input}

  Answer:
  ```

---

### Paradigm 2: Few-Shot In-Context Learning (3-Shot)
* **System Prompt**:
  ```text
  You are an expert AI assistant.
  Carefully review the provided demonstration examples to understand the expected format and task requirements.
  Solve the target task directly and accurately, following the demonstrated style.
  ```
* **User Template**:
  ```text
  Example 1
  Task: {demo_1_input}
  Answer: {demo_1_output}

  Example 2
  Task: {demo_2_input}
  Answer: {demo_2_output}

  Example 3
  Task: {demo_3_input}
  Answer: {demo_3_output}

  Target Task:
  {task_input}

  Answer:
  ```

---

### Paradigm 3: Zero-Shot Chain-of-Thought (Kojima et al.)
* **System Prompt**:
  ```text
  You are an expert AI reasoning assistant.
  Solve the given problem or task step by step, showing your logical deductions clearly.
  Conclude your reasoning with a clear and definite final answer.
  ```
* **User Template**:
  ```text
  {task_input}

  Let's think step by step.
  ```

---

### Paradigm 4: Short / Budget-Constrained Chain-of-Thought (Green AI)
* **System Prompt**:
  ```text
  You are an expert AI reasoning assistant.
  Solve the given task using at most 1 to 2 concise reasoning steps.
  Keep your explanation brief and focused, then state the final answer.
  ```
* **User Template**:
  ```text
  {task_input}

  Brief 1-2 step reasoning and answer:
  ```

---

### Paradigm 5: Long / Comprehensive Chain-of-Thought (Exhaustive Derivation)
* **System Prompt**:
  ```text
  You are an expert AI reasoning assistant.
  Provide a rigorous, detailed, and comprehensive step-by-step derivation.
  Explain all intermediate calculations, evidence, and logical deductions thoroughly before stating the final answer.
  ```
* **User Template**:
  ```text
  {task_input}

  Detailed step-by-step reasoning:
  ```

---

### Paradigm 6: Context-Grounded / RAG Prompting
* **System Prompt**:
  ```text
  You are a knowledge-grounded AI assistant.
  Answer the user's question or task strictly based on the provided reference context.
  If the context does not contain the answer, state that clearly. Be concise, factual, and accurate.
  ```
* **User Template**:
  ```text
  Context:
  {context_passage}

  Question / Task:
  {task_input}

  Answer:
  ```

---

## 3. Fixed In-Context Learning Exemplars (By Dataset)

To eliminate data leakage, few-shot demonstrations are extracted from independent training/validation splits and hardcoded into the deterministic registry:

### 3.1 Mathematical Reasoning (`GSM8K`)
* **Exemplar 1**:
  * *Problem*: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
  * *Answer*: `#### 72`
* **Exemplar 2**:
  * *Problem*: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
  * *Answer*: `#### 10`
* **Exemplar 3**:
  * *Problem*: "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"
  * *Answer*: `#### 5`

---

### 3.2 Open-Domain QA (`Natural Questions`)
* **Exemplar 1**:
  * *Question*: "who played the original darth vader in star wars?"
  * *Answer*: "David Prowse"
* **Exemplar 2**:
  * *Question*: "what is the currency used in switzerland?"
  * *Answer*: "Swiss franc"
* **Exemplar 3**:
  * *Question*: "when was the treaty of versailles signed?"
  * *Answer*: "June 28, 1919"

---

### 3.3 Long-Context Document QA (`ContextEval`)
* **Exemplar 1**:
  * *Document*: "The Apollo 11 spacecraft launched from Kennedy Space Center on July 16, 1969, carrying commander Neil Armstrong, command module pilot Michael Collins, and lunar module pilot Buzz Aldrin."
  * *Question*: "Who was the command module pilot on Apollo 11?"
  * *Answer*: "Michael Collins"
* **Exemplar 2**:
  * *Document*: "Photosynthesis occurs in two stages: the light-dependent reactions, which take place in the thylakoid membranes, and the light-independent Calvin cycle, which takes place in the stroma."
  * *Question*: "Where does the Calvin cycle take place in a plant cell?"
  * *Answer*: "Stroma"
* **Exemplar 3**:
  * *Document*: "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions, extending from the Arctic Ocean in the north to the Southern Ocean in the south."
  * *Question*: "Which ocean is the largest and deepest on Earth?"
  * *Answer*: "Pacific Ocean"

---

### 3.4 Text Summarization (`CNN/DailyMail`)
* **Exemplar 1**:
  * *Article*: "A rare blue diamond has sold at auction in Geneva for a record $48.5 million. The 12.03-carat 'Blue Moon' diamond was bought by a Hong Kong collector who immediately renamed it 'The Blue Moon of Josephine'..."
  * *Summary*: "A 12.03-carat blue diamond sold for $48.5 million at an auction in Geneva to a Hong Kong collector."
* **Exemplar 2**:
  * *Article*: "NASA's Curiosity rover has discovered evidence that a large lake once filled Gale Crater on Mars. Sedimentary rock layers indicate water persisted for millions of years..."
  * *Summary*: "NASA's Curiosity rover found sedimentary evidence that Gale Crater on Mars once hosted a long-standing lake."
* **Exemplar 3**:
  * *Article*: "Electric vehicle sales reached a historic milestone in Norway, accounting for more than 80% of all new passenger car sales last year. Government incentives..."
  * *Summary*: "Electric vehicles accounted for over 80% of new car sales in Norway, driven by strong government incentives."

---

## 4. LaTeX Appendix Table (For Conference Paper Submission)

```latex
\begin{table*}[t]
\centering
\small
\begin{tabular}{lp{6.8cm}p{5.5cm}c}
\toprule
\textbf{Prompting Paradigm} & \textbf{System Directive} & \textbf{User Template Wrapper} & \textbf{Demos} \\
\midrule
\textbf{Zero-Shot Direct} & Provide a direct, concise, and accurate answer without preamble. & \texttt{Task: \{input\} \textbackslash n Answer:} & 0 \\
\textbf{Few-Shot (3-Shot)} & Review demonstration examples to learn format and solve target task. & \texttt{Example 1...3 \textbackslash n Target: \{input\}} & 3 \\
\textbf{Zero-Shot CoT} & Solve the problem step by step, showing logical deductions clearly. & \texttt{\{input\} \textbackslash n Let's think step by step.} & 0 \\
\textbf{Short CoT} & Solve the task using at most 1--2 concise reasoning steps. & \texttt{\{input\} \textbackslash n Brief 1--2 step reasoning:} & 0 \\
\textbf{Long CoT} & Provide a rigorous, detailed, and comprehensive derivation. & \texttt{\{input\} \textbackslash n Detailed reasoning:} & 0 \\
\textbf{RAG / Context} & Answer strictly based on the provided reference context. & \texttt{Context: \{context\} \textbackslash n \{input\}} & 0 \\
\bottomrule
\end{tabular}
\caption{Generalized prompting paradigms, system directives, and operational templates evaluated across all task families in PromptEnergy-Bench.}
\label{tab:prompt_paradigms}
\end{table*}
```
