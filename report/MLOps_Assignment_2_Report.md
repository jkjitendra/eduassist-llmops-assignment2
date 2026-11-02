# MLOps Assignment 2 Report

## Objective and environment

EduAssist AI is a question-answering assistant, for academics. This experiment is to evaluate whether changes to prompt instructions affect response completeness, latency, token consumption, cost and output structure. Five Undergraduate level questions covering Supervised Learning, Overfitting, Gradient Descent, Precision and Recall, and Retrieval Augmented Generation were tested. The model was Gemini 3.5 Flash-Lite (gemini-3.5-flash-lite), accessed through the Gemini Developer API using google-genai SDK 2.34.0. The SDK uses API version v1 beta by default. Python notebooks executed through Google Colab connected to a Mac-based runtime; MLFlow 3.16.1 recorded prompt versions and experiments results on an external SSD.

## Task 1: API latency

For Task 1, Prompt v1 asked the model to answer each student question clearly. Request start and end timestamps, response statuses, elapsed latencies, and API-reported tokens were recorded. Four initial answers reached a 350-token generation limit and were therefore incomplete. Those questions were repeated using a 2,048-token limit; the already-complete Q2 result was retained. The final five answers all ended with STOP. Their mean latency was
2.4338 seconds, median 2.6815, minimum 1.3032, maximum 3.1746, and sample standard deviation 0.7021 seconds. Q3 was slowest, possibly reflecting answer length or variable network and server processing; causality was not established.

## Task 2: Token economics

For Task 2, the five completed v1 requests consumed 119 input tokens and 2,498 output tokens, totaling 2,617 tokens. Pricing was recorded on November 2, 2026, in US dollars: $0.30 per million input tokens and $2.50 per million output tokens under standard paid-tier pricing. The estimated paid-tier total for five requests was $0.0062807, averaging $0.00125614 per request. At 10,000 similar requests per month, the projected paid-tier cost is $12.56. The API project used an eligible free tier, so the estimated charge for these observed requests was $0; the projection is illustrative and does not imply that 10,000 requests fit within free quotas.

## Task 3: Prompt registration and evaluation

For Task 3, Prompt v1 was Basic, v2 specified concise undergraduate-friendly explanations, and v3 required JSON with answer, confidence, and key_points fields. All prompts were registered as versions of eduassist_academic_qa with meaningful commit messages and evaluated on the same five questions. Three separate MLflow runs stored the required averages for latency, input tokens, output tokens, estimated cost, and valid_json, plus schema validity and supplementary metrics. Response-level results and comparison CSVs were retained as artifacts. JSON syntax was checked by parsing each response, and schema validity was tested separately.

## Comparison of prompt versions

The comparison showed mean latency of 2.4338 seconds for v1, 1.7816 for v2, and 1.2783 for v3. Mean output tokens decreased from 499.6 to 292.0 to 142.6, while mean input tokens increased from 23.8 to 47.8 to 88.8 because the later prompts provided more instructions. Average estimated paid-tier request costs were $0.00125614, $0.00074434, and $0.00038314 respectively. Both v1 and v2 had 0% valid JSON in the five-response sample, while v3 achieved 100% JSON syntax and schema validity (five of five).

```
| Metric             | v1 Basic  | v2 Structured | v3 JSON   |
| ------------------ | --------- | ------------- | --------- |
| Avg. Latency       | 2.4338    | 1.7816        | 1.2783    |
| Avg. Input Tokens  | 23.8      | 47.8          | 88.8      |
| Avg. Output Tokens | 499.6     | 292.0         | 142.6     |
| Avg. Paid Cost     | $0.001256 | $0.000744     | $0.000383 |
| Valid JSON         | 0%        | 0%            | 100%      |
| Recommended        | No        | No            | Yes       |
```

## Production promotion and recommendation

Prompt v3 was selected for production because it met the JSON requirement and improved efficiency across 5 questions. All 5 v3 responses were valid JSON and passed the required schema checks for answer, confidence and key points. In contrast, v1 and v2 produced non valid JSON responses, so neither of them met the structured output requirement.

V3 recorded the lowest avg. latency at 1.2783 sec, compared with 1.7816 sec for v2 and 2.4338 sec for v1, which is about 47.5% lower than v1. Although v3 used more input tokens per request (88.8 vs 23.8 for v1), its concise responses reduced avg. output tokens from 499.6 to 142.6. This lowered estimated paid-tier cost per request from $0.00125614 to $0.00038314, approx. 69.5% less.

A cost-optimization recommendation is to reduce unnecessary output length while checking answer completeness. v3 received the MLflow Production alias, which was verified. These conclusions are based on 5 questions; JSON validity alone does not guarantee factual accuracy, so broader evaluation and human review are needed.
