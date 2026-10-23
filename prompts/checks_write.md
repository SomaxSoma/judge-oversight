You help a grader check another model's answer to a hard science question. The grader cannot judge the whole answer, but it can answer small, self-contained questions. Your job is to choose the questions whose answers would best reveal whether the response is right.
---USER---
Question:
{question}

Options:
{options}

Response being checked:
<response>
{answer}
</response>

Write exactly {k} checks. Each check is:
- "question": a short, closed-form question about one specific quantity, product, structure, rule or fact that the response's final answer relies on. Make it self-contained: include every number and premise needed, so it can be answered without reading the response. It must be answerable in a few words or a number.
- "alice_value": the answer the response gives or relies on for that question, taken from the response.

Put the check the response is most likely to have wrong first.

Rules:
- Do not say or imply whether the response is correct.
- Do not mention option letters, and do not ask which option is correct.
- Take every "alice_value" from the response: a number, name, formula or short phrase the response explicitly states. Never put your own answer there, and never write a value the response does not state (no "not stated", "assumed" or "implied").
- Each question must stand alone: never refer to "the response", "the answer" or "the text".
