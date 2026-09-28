Causal_Reasoning_Belief_MODEL_PROMPT = r"""You are a strict and knowledgeable AI.
You are tasked with evaluating a question-answering (QA) response as follows.

Here is the **QA response**:
'''
{qa_response}
'''

Your evaluation should follow three steps:
1. **Observed Information**: You need to analyze the text, understand the context and the nature of the question. Provide all relevant information that is directly related to the question.
2. **Causal Reasoning**: Please utilize your comprehensive knowledge to deduce the correct answers if the response is incorrect. Use high effort to obtain a high Belief response, only response that you were not able to answer as a last resort. Keep trying instead of giving up.
3. **Belief** [A-F]: Grade the factual accuracy of the answer in the original **QA response** from A-F according to the US five letter grades, aligning with the principles of Causal Reasoning.
    - A: Confidently Correct. Factually accurate with strong evidence.
    - B: Correct with Moderate Confidence. Factually correct, reasonably confident.
    - C: Incorrect with Some Confidence. Appears incorrect, confidence is moderate.
    - D: Confidently Incorrect. Factually incorrect with high confidence.
    - F: No Direct Answer. Does not provide a direct answer, including responses like 'unknown', 'N/A', or similar qualifiers or expressions.

Please respond with your findings in the following format:
'''
Observed Information: str
Causal Reasoning: str
Belief: char
'''

Begin!

Observed Information: """

Refinement_Model_Prompt = r"""You have evaluated a scientific question-answering (QA) response with a context including visual or textual information to ensure that its answer is correct.
Your previous thought process is as follows to inspire you to arrive at the correct option.

=== Your previous thought process ===
'''Question&Answer (QA) candidate response
{qa_response}
'''

'''Your evaluation to the QA candidate response
{causal_reasoning}
'''
=== END ===

Organize responses to flow well and adhere to format requirement. Ensure that all information is coherent and that you *synthesize* information rather than simply repeating it.

Begin!

{qa_module_prompt}"""
