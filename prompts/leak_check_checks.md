You audit checks written for a grader. Their author was forbidden from revealing a verdict or an answer, and every "response value" must come from the response itself. Flag only clear violations.
---USER---
Question:
{question}

Options:
{options}

The response being checked chose option ({alice_option}).

Response:
<response>
{answer}
</response>

Checks written for the grader:
{checks_block}

Check three things:
(a) states_verdict: do the checks state or clearly imply that the response is correct, or that it is incorrect?
(b) states_answer: do the checks state or clearly hint which option is correct, including by arguing for a different option?
(c) ungrounded: is any "response value" something the response does not say or rely on (for example, the author's own correction)?

Asking about a step, even one that may be wrong, is allowed and is not a violation.
