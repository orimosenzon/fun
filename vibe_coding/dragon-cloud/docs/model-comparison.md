# השוואת מודלי LLM סיניים בקוד פתוח — מרץ 2026

> עודכן לאחרונה: 2026-03-27. מקורות: HuggingFace, OpenRouter, Artificial Analysis, DeepWiki, ארכיטקטורות רשמיות.

---

## טבלת סקירה כללית

| מודל | ארגון | סה"כ פרמטרים | פרמטרים פעילים | חלון הקשר | רישיון | SWE-bench | GPQA-D | MMLU |
|---|---|---|---|---|---|---|---|---|
| **DeepSeek V3.2** | DeepSeek AI | 671B | 37B | 163K | MIT | ~73% | 80% | ~88% |
| **GLM-5** | Zhipu AI | 744B | 40B | 205K | MIT | 77.8% | ~87% | 92% |
| **Kimi K2.5** | Moonshot AI | ~1T | 32B | 256K | Modified MIT | 76.8% | 87.6% | 92% |
| **Qwen3.5-397B** | Alibaba | 397B | 17B | 262K | Apache 2.0 | 76.4% | ~88% | ~87% |
| **MiniMax M2.5** | MiniMax AI | 230B | 10B | 200K | Apache 2.0 | 80.2% | — | — |
| GPT-4o *(ייחוס)* | OpenAI | ~200B? | — | 128K | קנייני | ~33% | 53.6% | 87.2% |
| GPT-5 *(ייחוס)* | OpenAI | לא ידוע | — | 128K+ | קנייני | 80.0% | 92.4% | ~92% |
| Claude Opus 4.5 *(ייחוס)* | Anthropic | לא ידוע | — | 200K | קנייני | 80.9% | ~85% | ~89% |

---

## פירוט מפורט לכל מודל

### 1. DeepSeek V3.2
**שוחרר:** 1 ינואר 2026 | **ארכיטקטורה:** MoE + DeepSeek Sparse Attention (O(n) complexity)

| קוואנטיזציה | VRAM נדרש | GPU Setup | עלות ענן משוערכת/שעה |
|---|---|---|---|
| BF16 (מלא) | ~1,342 GB | 16–17× H100 80GB | לא מעשי |
| **FP8 (מומלץ)** | ~690 GB | **8× H100 80GB** | **~$16/hr** |
| FP8 + 128K ctx | ~1,128 GB | 8× H200 141GB | ~$28/hr |
| INT4 | ~350–400 GB | 4–5× H100 80GB | ~$8–10/hr |
| GGUF Q4_K_M | ~335 GB | CPU offload + multi-GPU | ניסיוני בלבד |

**עלות לטוקן (API מארוח):**
- Input: **$0.26/M tokens** | Output: **$0.38/M tokens** (OpenRouter)
- DeepSeek רשמי: $0.28 / $0.42

**throughput עצמאי:** ~50–80 tok/s ב-batch=1; ~2,000+ tok/s ב-batch גבוה על 8× H100

**frameworks:** vLLM ✓ | SGLang ✓ | TGI ✓ | Ollama ✓ | llama.cpp ✓

**HuggingFace:** `deepseek-ai/DeepSeek-V3.2`

---

### 2. GLM-5
**שוחרר:** 12 פברואר 2026 | **ארכיטקטורה:** MoE + DSA | **אומן על:** Huawei Ascend (ללא NVIDIA)

| קוואנטיזציה | VRAM / Storage | GPU Setup | עלות ענן משוערכת/שעה |
|---|---|---|---|
| BF16 (מלא) | ~1,488 GB | 20+ H100 80GB | לא מעשי |
| **FP8 (מומלץ)** | ~860 GB | **8× H200** או 8× B200 | **~$28/hr** |
| 8-bit GGUF | ~805 GB RAM | CPU/היברידי | איטי |
| **2-bit GGUF (Unsloth)** | ~241 GB disk | CPU בלבד אפשרי | ~$2–4/hr (CPU server) |
| 1-bit GGUF | ~176 GB disk | ניסיוני | ניסיוני |
| גודל על דיסק (מלא) | 1.65 TB | — | — |

**עלות לטוקן (API מארוח):**
- OpenRouter: **$0.72/M input** | **$2.30/M output**
- SiliconFlow: $1.00/M input | $3.20/M output

**ביצועים בולטים:**
- HumanEval: **99.0%** (שיאן בין כל המודלים)
- SWE-bench: **77.8%** (הטוב ביותר ב-open-source)
- AA Agentic Index: **#3 בעולם** (אחרי GPT-5 ו-Claude Opus 4.5 בלבד)
- AA Intelligence Index: **50** — הראשון ב-open-source לחצות את הסף הזה

**frameworks:** vLLM ✓ | SGLang ✓ | KTransformers ✓ | xLLM (Ascend) ✓ | Ollama ✓ | llama.cpp ✓

**HuggingFace:** `zai-org/GLM-5`

---

### 3. Kimi K2.5
**שוחרר:** 27 ינואר 2026 | **ארכיטקטורה:** MoE + MLA | **מולטימודאלי:** טקסט + ויז'ן | **אימון:** 15T+ טוקנים

| קוואנטיזציה | VRAM נדרש | GPU Setup | עלות ענן משוערכת/שעה |
|---|---|---|---|
| BF16 (מלא) | ~1,936 GB | 24+ H100 80GB | לא מעשי |
| **INT4 (רשמי)** | ~595 GB | **8× H100 80GB** | **~$16/hr** |
| INT4 (מומלץ) | ~630 GB | 8× H200 | ~$28/hr |
| GGUF Q4 | ~500 GB+ | CPU offload | ~10 tok/s על 24GB GPU + 256GB RAM |

**עלות לטוקן (API מארוח):**
- OpenRouter: **~$0.50/M input** | **~$2.50/M output** (משתנה)

**ביצועים בולטים:**
- AIME 2025: **96.1%**
- HLE (עם כלים): **50.2%** — מוביל בין כל המודלים
- MATH-500: **98.0%**
- Agent Swarm: מתזמן עד 100 sub-agents, מהיר פי 4.5 מ-Claude Opus 4.5 בעלות 76% נמוכה יותר

**frameworks:** vLLM ✓ | SGLang ✓ | TensorRT-LLM ✓ | Ollama ✓ | llama.cpp ✓

**HuggingFace:** `moonshotai/Kimi-K2.5`

---

### 4. Qwen3.5-397B-A17B
**שוחרר:** 16 פברואר 2026 | **ארכיטקטורה:** Hybrid MoE | **שפות:** 201 שפות | **מולטימודאלי:** טקסט + תמונה + וידאו

| קוואנטיזציה | VRAM נדרש | GPU Setup | עלות ענן משוערכת/שעה |
|---|---|---|---|
| BF16 (מלא) | ~794 GB | 10+ H100 80GB | ~$20/hr |
| **FP8 (מומלץ)** | ~400 GB | **5× H100 80GB** | **~$10/hr** |
| INT4 (4-bit) | ~200+ GB | 3× H100 / 4× RTX 4090 | ~$6–8/hr |
| GGUF Q4_K_M | ~230 GB | 3× A100 80GB | ~$6/hr |
| GGUF 2-bit | ~100 GB | 2× A100 80GB | ~$4/hr |
| Qwen3.5-32B (dense) | ~32 GB | **1× H100 80GB** | **~$2–3/hr** ← sweet spot |
| גרסאות קטנות (≤9B) | 5–6 GB | RTX 3060 12GB | מקומי בחינם |

**עלות לטוקן (API מארוח):**
- OpenRouter: **$0.39/M input** | **$2.34/M output**
- Alibaba רשמי (reasoning): $0.60/M | $3.60/M
- מהירות: **88 tok/s output** (מדד 8 מתוך 65 מודלים)

**ביצועים בולטים:**
- IFBench: **76.5%** — עובר GPT-5.2 (75.4%)
- LiveCodeBench v6: **83.6%** — מוביל open-source
- MathVision: **88.6%** — מוביל על vision math
- מהיר פי 8.6–19× מ-Qwen3-Max

⚠️ **באג ידוע:** Ollama ו-LM Studio שבורים עם 397B בגלל `presence_penalty`. השתמשו ב-vLLM, SGLang, או llama.cpp server.

**frameworks:** vLLM ✓ | SGLang ✓ | llama.cpp ✓ | TGI ✓ | Ollama ✗ (397B)

**HuggingFace:** `Qwen/Qwen3.5-397B-A17B` | `Qwen/Qwen3.5-397B-A17B-FP8`

---

### 5. MiniMax M2.5
**שוחרר:** פברואר 2026 | **ארכיטקטורה:** MoE | מהיר פי 2 ממודלים דומים

| קוואנטיזציה | זיכרון נדרש | GPU Setup | עלות ענן משוערכת/שעה |
|---|---|---|---|
| BF16 (מלא) | ~457 GB | 6+ H100 80GB | ~$12/hr |
| **vLLM production** | — | **4× 96GB GPU** | **~$8–10/hr** |
| GGUF UD-Q3_K_XL | ~101 GB disk / 96 GB RAM | 1× 16GB GPU + 96GB RAM | ~$2–4/hr |
| vLLM large context | — | 8× 144GB GPU | ~$28+/hr |

**עלות לטוקן (API מארוח):**
- SiliconFlow / automatio.ai: **~$0.15/M tokens** (input + output ביחד) — הזול ביותר!

**ביצועים בולטים:**
- SWE-bench: **80.2%** — כמעט זהה ל-Claude Opus 4.5 (80.9%)
- מחיר/ביצועים: **פי 20 זול יותר** מ-Claude Opus 4.5 על אותו SWE-bench

**frameworks:** vLLM ✓ | Ollama ✓ | llama.cpp ✓ | SGLang ✓

**HuggingFace:** `MiniMaxAI/MiniMax-M2.5`

---

## השוואת עלות ענן עצמאית (self-hosted)

| מודל | Hardware מינימום | עלות/שעה | tok/s (est.) | עלות פר M טוקן (output) |
|---|---|---|---|---|
| DeepSeek V3.2 | 8× H100 80GB | ~$16 | ~500 | **~$8.9** |
| GLM-5 | 8× H200 141GB | ~$28 | ~400 | **~$19.4** |
| Kimi K2.5 | 8× H100 80GB | ~$16 | ~400 | **~$11.1** |
| Qwen3.5-397B | 5× H100 80GB | ~$10 | ~600 | **~$4.6** |
| MiniMax M2.5 | 4× 96GB | ~$8 | ~700 | **~$3.2** |
| Qwen3.5-32B | 1× H100 80GB | ~$2.5 | ~1,500 | **~$0.46** ← כי-כלכלי |

> חישוב: עלות/שעה ÷ (tok/s × 3600) × 1,000,000

---

## סיכום המלצות לפי use case

| מקרה שימוש | מודל מומלץ | סיבה |
|---|---|---|
| **מחיר/ביצועים** | MiniMax M2.5 | $0.15/M, SWE-bench כמו Claude Opus |
| **Agentic / tool-use** | GLM-5 | #3 בעולם ב-Agentic Index |
| **Reasoning / math** | Kimi K2.5 | AIME 96.1%, HLE 50.2% |
| **Instruction following** | Qwen3.5-397B | עובר GPT-5.2 ב-IFBench |
| **Coding + ecosystem** | DeepSeek V3.2 | MIT, הכי בשל ב-deployment |
| **חומרה מוגבלת** | Qwen3.5-32B | שווה ל-GPT-4o, רץ על H100 אחד |
| **CPU בלבד (ניסיוני)** | GLM-5 2-bit GGUF | ~241GB RAM, ללא GPU |

---

## מקורות
- HuggingFace model cards: deepseek-ai, zai-org, moonshotai, Qwen, MiniMaxAI
- OpenRouter pricing: openrouter.ai
- Artificial Analysis: artificialanalysis.ai
- DeepWiki: deepwiki.com/zai-org/GLM-5
- Novita Blog: kimi-k2-5-vram-requirements
- hardware-corner.net: qwen3-hardware-requirements
- Spheron GPU Cheat Sheet 2026
