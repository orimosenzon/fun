# dragon-cloud — פריסת LLM סיני בקוד פתוח על קלאסטר ענן

## מטרת הפרויקט
הורדה, הגדרה והרצה של מודל LLM סיני בקוד פתוח על קלאסטר GPU בענן.
המטרה: inference מהיר, עלות-תועלת, ובקרה מלאה על המודל.

## מצב נכון ל-2026-03-27

### המודלים המובילים (נכון למרץ 2026)
| מודל | ארגון | פרמטרים | בולט ב |
|---|---|---|---|
| GLM-5 | Zhipu AI | 744B (40B פעיל, MoE) | #1 בלוח — coding, math, הלוצינציות נמוכות. אומן על Huawei ללא NVIDIA |
| Kimi K2.5 | Moonshot AI | 1T (32B פעיל) | HumanEval 99.0, multimodal, agent swarm |
| Qwen3.5 | Alibaba | 397B (17B פעיל) | Agentic, tool-use, מהיר מאוד |
| DeepSeek V3.2 | DeepSeek | — | Coding, reasoning, חסכוני |

### שאלות פתוחות לסשן הבא
- איזה מודל לבחור? (GLM-5 / Kimi K2.5 / Qwen3.5 / DeepSeek V3.2)
- על איזה ענן? (AWS / GCP / Azure / RunPod / Lambda Labs)
- inference בלבד או גם fine-tuning?
- סקייל: גרסת distilled קטנה או מודל מלא?
- פלט רצוי: סקריפט הרצה? API? ממשק?

## סטאק (טנטטיבי)
- Python
- vLLM / TGI (Text Generation Inference) להרצה
- HuggingFace Hub להורדת המודל
- Terraform / Ansible לתשתית הענן (TBD)

## שם הפרויקט
dragon-cloud — דרקון (לייצוג המודל הסיני) + ענן (cloud deployment)
