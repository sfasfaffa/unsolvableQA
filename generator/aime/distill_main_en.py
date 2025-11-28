import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import openai
import os
import textwrap
import re
import json
import time

# --- Environment Variables deepseek---
os.environ["OPENAI_API_KEY"] = "sk-4093ceb1897e49e0bb1fbc6a8d754dab"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

# sk-tPYvRiu9xrOT4kyJD493194f599d4aB7912771838f9f56E7
# os.environ["OPENAI_API_KEY"] = "sk-tPYvRiu9xrOT4kyJD493194f599d4aB7912771838f9f56E7"
# os.environ["OPENAI_BASE_URL"] = "https://one.ooo.cool/v1"

# --- Contradiction Types ---
CONTRADICTION_TYPES = {
    "Constraint Contradiction": {
        "description": "Construct mutually conflicting mathematical conditions by adding or modifying constraints.",
        "strategy": "External Chain Intersection"
    },
    "Axiom Contradiction": {
        "description": "Construct scenarios that violate basic mathematical axioms by modifying reasoning nodes.", 
        "strategy": "Internal Node Disruption"
    }
}

def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

class UnsolvableProblemGenerator:
    def __init__(self, model_name="deepseek-chat", final_model_name="deepseek-chat"):
        print(f"--- Initializing model: {model_name} ---")
        try:
            self.client = openai.OpenAI()
            self.model_name = model_name
            # model to use for final verification / contradiction detection
            self.final_model_name = final_model_name
            print("--- Model client initialized successfully ---\n")
        except Exception as e:
            print(f"Error: Unable to initialize OpenAI client.")
            raise

    def _call_llm(self, prompt, temperature=0.5, is_json=False, model_name='deepseek-chat'):
        try:
            response_format = {"type": "json_object"} if is_json else {"type": "text"}
            # allow overriding the model for specific calls (e.g. final verification)
            model_to_use = model_name if model_name is not None else self.model_name
            if model_to_use == "deepseek-reasoner":
                print(f"--- Using model: {model_to_use} ---")
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=32000
                    # max_tokens=22000
                )
                return response.choices[0].message.content
            else:
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=8192
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling model API: {e}")
            return None
    
    def _call_llm_with_retries(self, prompt, temperature=0.5, is_json=False, model_name='deepseek-chat', max_attempts=2, backoff_seconds=1.0):
        """Call the LLM with a small retry loop when no response (None or empty) is returned.

        Retries up to max_attempts times (including the first attempt). Returns the first non-empty response or None.
        """
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            resp = self._call_llm(prompt, temperature=temperature, is_json=is_json, model_name=model_name)
            if resp:
                if isinstance(resp, str) and resp.strip() == "":
                    resp = None
            if resp:
                if attempt > 1:
                    print(f"[INFO] LLM succeeded on attempt {attempt}/{max_attempts}")
                return resp
            else:
                print(f"[WARN] LLM returned no response on attempt {attempt}/{max_attempts}.")
                if attempt < max_attempts:
                    time.sleep(backoff_seconds)
        print(f"[ERROR] LLM failed after {max_attempts} attempts.")
        return None
            
    def _extract_final_answer(self, response_text: str) -> str:
        # 1) Try \boxed{...}
        boxed_match = re.search(r"\\boxed\{(.+?)\}", response_text)
        if boxed_match:
            ans = boxed_match.group(1).strip()
            while ans.endswith('}') or ans.endswith(' '):
                ans = ans.rstrip('} ').strip()
            return ans
        # 2) Try common "Answer:" markers
        ans_match = re.search(r"Answer[:\s]*\$?([^\n\r]+)\$?", response_text, flags=re.IGNORECASE)
        if ans_match:
            return ans_match.group(1).strip()
        # 3) Try lines that start with a boxed-like or final token, otherwise take last non-empty line
        lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
        if not lines:
            return "Failed to extract answer"
        # Prefer a short last line that looks like an answer (digits, fraction, simple expression)
        for line in reversed(lines[-4:]):
            if re.match(r"^[\[\(\-]?[0-9+\-*/^.()\\s\\$\\,]+[\]\)]?$", line):
                return line
        # fallback: return last line
        return lines[-1]

    def _math_equiv(self, ans1, ans2):
        def clean(expr):
            return str(expr).replace('$','').replace('\\left','').replace('\\right','').strip()
        a1 = clean(ans1)
        a2 = clean(ans2)
        # Try symbolic comparison
        try:
            transformations = (standard_transformations + (implicit_multiplication_application,))
            e1 = parse_expr(a1, transformations=transformations, evaluate=True)
            e2 = parse_expr(a2, transformations=transformations, evaluate=True)
            diff = sympy.simplify(e1 - e2)
            if diff == 0:
                return True
        except Exception:
            pass
        # Try numeric comparison if both parseable to numbers
        try:
            v1 = float(sympy.N(parse_expr(a1)))
            v2 = float(sympy.N(parse_expr(a2)))
            if abs(v1 - v2) < 1e-6:
                return True
        except Exception:
            pass
        # final fallback: compare normalized strings
        try:
            return a1 == a2
        except Exception:
            return False

    def _suggest_changes_for_unsolvable(self, solvable_problem, cot_a, contradiction_plan, current_unsolvable_problem, max_suggestions=3):
        """
        Ask the strong reasoner to produce a concise change suggestion (1 line or up to 3 short bullets)
        that would make the CURRENT_UNSOLVABLE_PROBLEM more likely to exhibit the intended late-stage
        contradiction described in contradiction_plan. Return a dict with keys:
          - suggestions: list of short suggestion strings
          - rationale: short rationale string

        The function is robust: tries to parse JSON, falls back to simple line extraction.
        """
        try:
            prompt = (
                "You are a concise and pragmatic mathematical editor.\n\n"
                "We give you:\n"
                "1) ORIGINAL PROBLEM:\n" + solvable_problem + "\n\n"
                "2) ORIGINAL CoT_A:\n" + cot_a + "\n\n"
                "3) CONTRADICTION PLAN (JSON):\n" + json.dumps(contradiction_plan, ensure_ascii=False, indent=2) + "\n\n"
                "4) THE CURRENT GENERATED UNSOLVABLE PROBLEM:\n" + current_unsolvable_problem + "\n\n"
                "Task:\n"
                "- Provide 1 to 3 very short, concrete, and minimal change suggestions (each 1 sentence or less) that would likely cause the intended late-stage contradiction to appear when the problem is solved.\n"
                "- Output MUST be a single-line JSON object and nothing else, exactly like this example:\n"
                "  {\"suggestions\": [\"Change X to Y\", \"Also ...\"], \"rationale\": \"One-line rationale\"}\n\n"
                "Be concise: suggestions should target the generated problem text (e.g., change a coefficient, add a constraint, alter a sign, add/omit a clause).\n"
                "Do NOT include explanatory paragraphs or anything outside the one-line JSON object.\n"
            )
            resp = self._call_llm_with_retries(prompt, temperature=0.0, model_name="deepseek-reasoner", max_attempts=2, backoff_seconds=1.0)
            if not resp:
                return None
            # Try parse JSON from response
            text = resp.strip()
            # Allow if model prints code fences or trailing commentary - extract first JSON-looking substring
            m = re.search(r"\{[\s\S]*\}", text)
            json_text = m.group(0) if m else text
            try:
                parsed = json.loads(json_text)
                suggestions = parsed.get("suggestions") or parsed.get("suggestion") or []
                if isinstance(suggestions, str):
                    suggestions = [suggestions]
                rationale = parsed.get("rationale", "")
                return {"suggestions": suggestions, "rationale": rationale}
            except Exception:
                # Fallback: take up to max_suggestions non-empty lines from the response
                lines = [l.strip("- ") for l in text.splitlines() if l.strip()]
                if not lines:
                    return None
                # take first up to max_suggestions lines as suggestions
                suggs = lines[:max_suggestions]
                rationale = suggs[0] if suggs else ""
                return {"suggestions": suggs, "rationale": rationale}
        except Exception as e:
            print(f"[DEBUG] _suggest_changes_for_unsolvable failed: {e}")
            return None

    def step1_check_solvability_and_get_cot(self, solvable_problem, ground_truth_answer):
        print("### Step 1: Check base capability and extract original chain of thought (CoT_A) ###")
        prompt = (
            "You are a top logical reasoning expert. Solve the following problem.\n"
            "If possible, include a brief chain-of-thought and a clear final answer (boxed or labeled 'Answer:').\n"
            "Prefer a concise final line containing the numeric or symbolic answer.\n\n"
            f"--- Problem Start ---\n{solvable_problem}\n--- Problem End ---"
        )
        # Use the stronger reasoner for initial capability check / CoT generation
        response_text = self._call_llm_with_retries(prompt, model_name="deepseek-reasoner", max_attempts=2, backoff_seconds=1.0)
        if not response_text:
            print("❌ Model failed: no response from LLM.")
            return None
        # Debug: show start of response for inspection
        print("LLM response preview:\n", response_text[:800])
        # Try to extract final answer
        llm_final_answer = self._extract_final_answer(response_text)
        gt = str(ground_truth_answer).strip()
        while gt.endswith('}') or gt.endswith(' '):
            gt = gt.rstrip('} ').strip()
        if llm_final_answer and llm_final_answer != "Failed to extract answer":
            if self._math_equiv(llm_final_answer, gt):
                print(f"✅ Model capability check passed (Model answer: '{llm_final_answer}', Correct answer: '{gt}').")
                return response_text
            else:
                print(f"❌ Model failed capability check (Model answer: '{llm_final_answer}' != Ground truth: '{gt}'). Aborting pipeline.")
                return None
        else:
            print("❌ Could not reliably extract an answer from LLM response. Aborting pipeline.")
            return None

    def _generate_constraint_contradiction_v0(self, original_cot, difficulty):
        print("### Constructing Constraint Contradiction: External Chain Intersection ###")
        prompt = (
            f"You are a math problem design expert specializing in problems with logical contradictions. Current difficulty level: {difficulty}/10. Higher difficulty means more subtle and complex contradictions.\n\n"
            "--- Original Chain of Thought ---\n"
            f"{original_cot}\n\n"
            "--- Your Task: Construct a Constraint Contradiction ---\n"
            "1. Analyze the chain of thought and identify key nodes where parallel external conditions can be added.\n"
            "2. Design an external reasoning path that intersects with the original chain at a key point and creates a conflict.\n"
            "3. The conflict should be: two seemingly reasonable conditions cannot both hold at the intersection.\n"
            f"4. Adjust the subtlety and complexity of the contradiction according to difficulty level {difficulty}/10.\n\n"
            "Return in JSON format:\n"
            "{\n"
            '  "strategy": "External Chain Intersection",\n'
            '  "intersection_analysis": "Explain which step is chosen for intersection.",\n'
            '  "external_condition": "The new constraint to add.",\n'
            '  "contradiction_description": "Why these conditions conflict at the intersection.",\n'
            '  "problem_modification_suggestion": "How to modify the original problem based on this contradiction."\n'
            "}"
            )
        response = self._call_llm_with_retries(prompt, temperature=0.7, is_json=True, model_name="deepseek-chat", max_attempts=2, backoff_seconds=1.0)
        if not response:
            return None

        # Be robust: _call_llm may return a dict when is_json=True or a string containing JSON
        constraint_plan = None
        try:
            if isinstance(response, dict):
                constraint_plan = response
            else:
                # try direct JSON parse
                try:
                    constraint_plan = json.loads(response)
                except Exception:
                    # try to extract the first JSON object from the text
                    m = re.search(r"\{[\s\S]*\}", str(response))
                    if m:
                        try:
                            constraint_plan = json.loads(m.group(0))
                        except Exception:
                            constraint_plan = None

            if not constraint_plan:
                # Ask the model to re-output only the JSON object (strict)
                followup = (
                    "The previous response could not be parsed as valid JSON.\n"
                    "Please REPLY WITH ONLY the JSON object described previously and nothing else.\n\n"
                    + prompt
                )
                follow = self._call_llm_with_retries(followup, temperature=0.0, is_json=True, max_attempts=2, backoff_seconds=0.5)
                if isinstance(follow, dict):
                    constraint_plan = follow
                else:
                    if follow:
                        try:
                            constraint_plan = json.loads(follow)
                        except Exception:
                            m2 = re.search(r"\{[\s\S]*\}", str(follow))
                            if m2:
                                try:
                                    constraint_plan = json.loads(m2.group(0))
                                except Exception:
                                    constraint_plan = None
                    else:
                        constraint_plan = None

            if constraint_plan:
                print("✅ Constraint contradiction plan generated successfully")
                return constraint_plan
            else:
                print("❌ Failed to parse constraint contradiction plan")
                return None
        except Exception:
            print("❌ Failed to parse constraint contradiction plan")
            return None



    def _generate_constraint_contradiction(self, original_cot, difficulty):
        print("### Constructing Constraint Contradiction: External Chain Intersection ###")
        prompt = (
            f"You are a senior math problem designer. Construct a subtle Constraint Contradiction aimed to be hard to spot: prefer placing the contradictory condition near the END of the original chain-of-thought so it appears only after most reasoning steps. Current difficulty level: {difficulty}/10 — higher difficulty means more hidden and late-appearing contradictions.\n\n"
            "--- Original Chain of Thought ---\n"
            f"{original_cot}\n\n"
            "--- Your Task: Construct a Subtle, End-of-Chain Constraint Contradiction ---\n"
            "1. Read the original chain-of-thought and identify a late-stage step (near the conclusion) where a small external constraint can be added to produce a logical conflict only when the chain reaches that stage.\n"
            "2. The added constraint should be plausible in isolation and should not contradict early steps; it should produce a contradiction only when combined with the final reasoning steps.\n"
            "3. Keep the contradiction concise and minimal: prefer 1-2 lines that introduce the conflicting condition at the end.\n"
            "4. After drafting the contradiction, ANALYZE the modified chain: simulate the reasoning forward to confirm the contradiction appears at the intended late step. Include a short verification note explaining why it is subtle and which late step it triggers.\n\n"
            "Return in JSON format with the following fields:\n"
            "{\n"
            '  "strategy": "External Chain Intersection (subtle_end)",\n'
            '  "chosen_intersection_step": "Which late step (description) is targeted for intersection.",\n'
            '  "external_condition": "The small external constraint to add (one or two sentences).",\n'
            '  "contradiction_description": "Why the condition conflicts but only at the late step.",\n'
            '  "verification_note": "A brief simulation / check showing that the contradiction indeed appears at the targeted late step.",\n'
            '  "problem_modification_suggestion": "How to modify the original problem text to include the external condition."\n'
            "}"
            )
        response = self._call_llm_with_retries(prompt, temperature=0.7, is_json=True, model_name="deepseek-reasoner", max_attempts=2, backoff_seconds=1.0)
        if not response:
            return None

        # Be robust: _call_llm may return a dict when is_json=True or a string containing JSON
        constraint_plan = None
        try:
            if isinstance(response, dict):
                constraint_plan = response
            else:
                # try direct JSON parse
                try:
                    constraint_plan = json.loads(response)
                except Exception:
                    # try to extract the first JSON object from the text
                    m = re.search(r"\{[\s\S]*\}", str(response))
                    if m:
                        try:
                            constraint_plan = json.loads(m.group(0))
                        except Exception:
                            constraint_plan = None

            if not constraint_plan:
                # Ask the model to re-output only the JSON object (strict)
                followup = (
                    "The previous response could not be parsed as valid JSON.\n"
                    "Please REPLY WITH ONLY the JSON object described previously and nothing else.\n\n"
                    + prompt
                )
                follow = self._call_llm_with_retries(followup, temperature=0.0, is_json=True, max_attempts=2, backoff_seconds=0.5)
                if isinstance(follow, dict):
                    constraint_plan = follow
                else:
                    if follow:
                        try:
                            constraint_plan = json.loads(follow)
                        except Exception:
                            m2 = re.search(r"\{[\s\S]*\}", str(follow))
                            if m2:
                                try:
                                    constraint_plan = json.loads(m2.group(0))
                                except Exception:
                                    constraint_plan = None
                    else:
                        constraint_plan = None
            if constraint_plan:
                print("✅ Constraint contradiction plan generated successfully")
                return constraint_plan
            else:
                print("❌ Failed to parse constraint contradiction plan")
                return None
        except Exception:
            print("❌ Failed to parse constraint contradiction plan")
            return None

    def _generate_axiom_contradiction(self, original_cot, difficulty):
        print("### Constructing Axiom Contradiction: Internal Node Disruption ###")
        prompt = (
            f"You are a senior math problem designer. Construct a subtle Axiom Contradiction that alters a late-stage reasoning node so the violation appears only in the concluding steps. Current difficulty level: {difficulty}/10 — higher difficulty means more hidden and late-appearing violations.\n\n"
            "--- Original Chain of Thought ---\n"
            f"{original_cot}\n\n"
            "--- Your Task: Construct a Subtle, Late-Stage Axiom Contradiction ---\n"
            "1. Inspect the chain-of-thought and identify a concluding node whose correctness depends on a core axiom or definition.\n"
            "2. Modify that node's assumption or step minimally so it violates an axiom only during the final reasoning, while early steps remain consistent.\n"
            "3. Keep the modification small and plausible (a single altered relation, sign, or property) so the contradiction is subtle.\n"
            "4. After drafting the modification, ANALYZE the resulting chain: simulate forward to confirm the axiom violation triggers only at the end. Provide a short verification note.\n\n"
            "Return in JSON format with these fields:\n"
            "{\n"
            '  "strategy": "Internal Node Disruption (subtle_end)",\n'
            '  "target_node_description": "Which late node is modified (description).",\n'
            '  "original_reasoning": "The original correct step content.",\n'
            '  "modified_reasoning": "The minimally modified step that violates an axiom.",\n'
            '  "violated_axiom": "Which axiom or definition is violated (brief).",\n'
            '  "verification_note": "A brief simulation / check showing the violation appears at the targeted late step.",\n'
            '  "problem_modification_suggestion": "How to change the original problem text to introduce the modified reasoning node."\n'
            "}"
            )
        response = self._call_llm_with_retries(prompt, temperature=0.7, is_json=True, model_name="deepseek-reasoner", max_attempts=2, backoff_seconds=1.0)
        if not response:
            return None

        # Robust parsing similar to constraint case
        axiom_plan = None
        try:
            if isinstance(response, dict):
                axiom_plan = response
            else:
                try:
                    axiom_plan = json.loads(response)
                except Exception:
                    m = re.search(r"\{[\s\S]*\}", str(response))
                    if m:
                        try:
                            axiom_plan = json.loads(m.group(0))
                        except Exception:
                            axiom_plan = None
            if not axiom_plan:
                followup = (
                    "The previous response could not be parsed as valid JSON.\n"
                    "Please REPLY WITH ONLY the JSON object described previously and nothing else.\n\n"
                    + prompt
                )
                follow = self._call_llm_with_retries(followup, temperature=0.0, is_json=True, model_name="deepseek-chat", max_attempts=2, backoff_seconds=0.5)
                if isinstance(follow, dict):
                    axiom_plan = follow
                else:
                    if follow:
                        try:
                            axiom_plan = json.loads(follow)
                        except Exception:
                            m2 = re.search(r"\{[\s\S]*\}", str(follow))
                            if m2:
                                try:
                                    axiom_plan = json.loads(m2.group(0))
                                except Exception:
                                    axiom_plan = None
                    else:
                        axiom_plan = None

            if axiom_plan:
                print("✅ Axiom contradiction plan generated successfully")
                return axiom_plan
            else:
                print("❌ Failed to parse axiom contradiction plan")
                return None
        except Exception:
            print("❌ Failed to parse axiom contradiction plan")
            return None

    def _generate_unsolvable_problem(self, solvable_problem, contradiction_plan, contradiction_type):
        print(f"### Generating unsolvable problem based on {contradiction_type} plan (English version) ###")
        if contradiction_type == "Constraint Contradiction":
            prompt = (
                "You are a careful math problem designer. Using the provided contradiction plan, produce a modified problem that embeds the contradiction subtly (preferably the condition should only reveal the contradiction when the solver reaches the final reasoning steps).\n\n"
                f"--- Original Problem ---\n{solvable_problem}\n\n"
                f"--- Constraint Contradiction Plan (with verification) ---\n{json.dumps(contradiction_plan, ensure_ascii=False, indent=2)}\n\n"
                "Generation requirements:\n"
                "- Embed the contradictory condition in the problem text such that it appears plausible but causes a logical conflict when combined with late-stage reasoning.\n"
                "- The modified problem description must explicitly include the contradictory condition(s) (numbered). Keep the wording compact.\n"
                "- During generation, SIMULATE the solver's reasoning to verify the contradiction appears at the intended late step and include a one-line generation-time verification comment (kept as an inline comment prefixed by '#GEN_VERIFY:').\n"
                "- Do NOT include long explanations; output only the modified problem text followed by the single-line '#GEN_VERIFY:' comment."
            )
        else:  # Axiom Contradiction
            prompt = (
                "You are a careful math problem designer. Using the provided axiom-violation plan, produce a modified problem that introduces a minimal modification to a late-stage node so that an axiom is violated only at the conclusion.\n\n"
                f"--- Original Problem ---\n{solvable_problem}\n\n"
                f"--- Axiom Contradiction Plan (with verification) ---\n{json.dumps(contradiction_plan, ensure_ascii=False, indent=2)}\n\n"
                "Generation requirements:\n"
                "- Modify the problem text to introduce the minimally changed condition (single-line change preferred) that causes an axiom violation at the end of reasoning.\n"
                "- During generation, SIMULATE the solver's final steps to confirm the axiom violation occurs and include a one-line generation-time verification comment prefixed by '#GEN_VERIFY:'.\n"
                "- Output only the modified problem text followed by the single-line '#GEN_VERIFY:' comment; avoid long explanations."
            )
        response = self._call_llm_with_retries(prompt, temperature=0.3, model_name="deepseek-reasoner", max_attempts=2, backoff_seconds=1.0)
        # The response is expected to be the problem text followed by a one-line '#GEN_VERIFY:' comment.
        return response.strip() if response else None

    def _improve_unsolvable_problem(self, solvable_problem, cot_a, contradiction_plan, current_unsolvable_problem):
        """Ask the reasoner to propose an improved unsolvable problem if contradiction verification failed.

        Returns improved problem text (with a '#IMPROVED_GEN_VERIFY:' line) or None.
        """
        print("### Fallback: Generating improved unsolvable problem variant ###")
        prompt = (
            "You attempted to construct an unsolvable math problem but both verification passes failed to conclusively detect a contradiction. \n"
            "We will give you the full construction context again. Your task: REFINE or REPLACE the modified problem so that a LATE-STAGE contradiction becomes CLEAR and UNAVOIDABLE when solving.\n\n"
            "OUTPUT REQUIREMENTS (STRICT):\n"
            "- Produce ONLY the improved problem statement followed by ONE single-line comment starting with '#IMPROVED_GEN_VERIFY:' summarizing the intended late-step contradiction.\n"
            "- Keep wording concise; explicitly embed the conflicting conditions (number or bullet them if helpful).\n"
            "- The contradiction should trigger only after most reasoning steps (late-stage) but be decisive.\n"
            "- Do NOT include solution steps, analysis paragraphs, JSON, or extra commentary.\n\n"
            f"--- ORIGINAL PROBLEM ---\n{solvable_problem}\n\n"
            f"--- ORIGINAL CoT_A (reference chain) ---\n{cot_a}\n\n"
            f"--- ORIGINAL CONTRADICTION PLAN ---\n{json.dumps(contradiction_plan, ensure_ascii=False, indent=2)}\n\n"
            f"--- PREVIOUS UNSOLVABLE PROBLEM (failed verification) ---\n{current_unsolvable_problem}\n\n"
            "Now produce the improved unsolvable problem + one '#IMPROVED_GEN_VERIFY:' line."
        )
        improved = self._call_llm_with_retries(prompt, temperature=0.2, model_name="deepseek-reasoner", max_attempts=2, backoff_seconds=1.0)
        if not improved:
            print("❌ Fallback improvement attempt returned no response.")
            return None
        return improved.strip()

    def _solve_and_detect_contradiction(self, unsolvable_problem):
        print("### Attempting to solve new problem and detect contradiction (English version) ###")
        prompt = (
            "You are a rigorous mathematician. Analyze the following problem carefully.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Carefully check if the problem contains any logical contradictions or conflicting conditions.\n"
            "2. If you find ANY contradiction or conflict (even subtle ones), you MUST immediately conclude the problem is unsolvable.\n"
            "3. Do NOT attempt to force a solution if contradictions exist.\n"
            "4. Do NOT ignore or work around contradictions.\n\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "[Analysis]\n"
            "Analyze each condition and check for contradictions step by step.\n\n"
            "[Conclusion]\n"
            "- If contradictions found: Output EXACTLY \\boxed{Unsolvable} and explain the contradiction.\n"
            "- If no contradictions: Output \\boxed{your_answer} with the solution.\n\n"
            f"--- Problem Start ---\n{unsolvable_problem}\n--- Problem End ---"
        )
        # Use the stronger reasoner to attempt solving the generated unsolvable problem
        response = self._call_llm_with_retries(prompt, temperature=0.1, model_name="deepseek-reasoner", max_attempts=2, backoff_seconds=1.0)
        if not response:
            return None, False

        contradiction_detected = False
        match_reason = "none"

        # Simplified detection: only check boxed conclusion
        # Keywords indicating unsolvability
        keywords = [
            "contradiction", "unsolvable", "no solution", "no solutions",
            "conflict", "impossible", "inconsistent", "illogical", "无解", "矛盾", "没有解"
        ]
        # Check if \boxed{...} contains unsolvability keywords
        boxed_match = re.search(r"\\boxed\{([^}]*)\}", response, flags=re.IGNORECASE)
        if boxed_match:
            boxed_text = boxed_match.group(1).lower()
            for kw in keywords:
                if kw in boxed_text:
                    contradiction_detected = True
                    match_reason = f"boxed: {boxed_match.group(1).strip()}"
                    print(f"[DEBUG] Contradiction detected by boxed conclusion: {boxed_match.group(1).strip()}")
                    break

        # If boxed check didn't detect contradiction, call LLM verifier
        if not contradiction_detected:
            try:
                verifier = self._verify_with_llm(response)
                v = str(verifier.get("verdict", "UNKNOWN")).upper()
                if v == "UNSOLVABLE":
                    contradiction_detected = True
                    match_reason = f"llm_verifier:{verifier.get('confidence', 0.0)}"
                    print(f"[DEBUG] LLM verifier accepted as UNSOLVABLE: {verifier}")
                else:
                    print(f"[DEBUG] LLM verifier result: {verifier}")
            except Exception as e:
                print(f"[DEBUG] verifier call failed: {e}")

        if contradiction_detected:
            print(f"[DEBUG] Contradiction detected by: {match_reason}")

        return response, contradiction_detected

    def _verify_with_llm(self, solver_response, max_tail_chars=800):
        """Ask an LLM to judge whether the solver_response implies the problem is solvable or unsolvable.

        Returns a dict with keys: verdict ('UNSOLVABLE'|'SOLVABLE'|'UNKNOWN'), confidence (0-1 float, optional), reason (str).
        """
        try:
            tail = solver_response.strip()[-max_tail_chars:]
        except Exception:
            tail = solver_response

        prompt = (
            "You are an impartial verifier. Given the following model solution attempt (or analysis), decide whether the problem described is solvable or unsolvable. "
            "Return a JSON object with fields: verdict (one of 'UNSOLVABLE','SOLVABLE','UNKNOWN'), confidence (0-1 float), and reason (short text).\n\n"
            "Do not include extra commentary outside the JSON. Use 'UNSOLVABLE' when the response clearly demonstrates a contradiction or states the problem is unsolvable; 'SOLVABLE' when a valid solution is provided; otherwise 'UNKNOWN'.\n\n"
            "Solution attempt (last portion):\n" + tail
        )

        # Use the final model for verification if available. Request JSON output and require parseable JSON.
        # Verifier should remain the chat model (deepseek-chat)
        verifier_resp = self._call_llm_with_retries(prompt, temperature=0.0, is_json=True, model_name="deepseek-chat", max_attempts=2, backoff_seconds=0.5)
        if not verifier_resp:
            return {"verdict": "UNKNOWN", "confidence": 0.0, "reason": "no verifier response"}

        # Expect strict JSON from the verifier. If parsing fails or the JSON lacks a clear verdict, return UNKNOWN.
        try:
            # verifier_resp may already be a dict/object when the client returns JSON; handle both cases
            if isinstance(verifier_resp, dict):
                parsed = verifier_resp
            else:
                parsed = json.loads(verifier_resp)
            v = parsed.get("verdict") or parsed.get("Verdict") or parsed.get("result")
            if isinstance(v, str):
                verdict = v.strip().upper()
            else:
                verdict = str(v).upper() if v is not None else "UNKNOWN"
            confidence = parsed.get("confidence") or parsed.get("score") or 0.0
            reason = parsed.get("reason") or parsed.get("explanation") or ""
            return {"verdict": verdict, "confidence": float(confidence) if confidence is not None else 0.0, "reason": reason}
        except Exception:
            # Do not attempt unreliable text/keyword fallback — the verifier must return valid JSON.
            return {"verdict": "UNKNOWN", "confidence": 0.0, "reason": "invalid_verifier_output"}
    def run_pipeline(self, config, max_retries=3):
        solvable_problem = config["solvable_problem"]
        ground_truth_answer = config["ground_truth_answer"]
        # Randomly choose contradiction type: 60% Constraint, 40% Axiom
        import random
        contradiction_type = random.choices(
            ["Constraint Contradiction", "Axiom Contradiction"],
            weights=[0.6, 0.4],
            k=1
        )[0]
        print(f"[INFO] Randomly selected contradiction type: {contradiction_type}")
        difficulty = config.get("difficulty", 5)

        cot_a = self.step1_check_solvability_and_get_cot(solvable_problem, ground_truth_answer)
        if not cot_a:
            return {"status": "failed_at_base_problem"}
        # Generate contradiction plan
        if contradiction_type == "Constraint Contradiction":
            contradiction_plan = self._generate_constraint_contradiction(cot_a, difficulty)
        else:
            contradiction_plan = self._generate_axiom_contradiction(cot_a, difficulty)
        
        if not contradiction_plan:
            return {"status": "failed_to_generate_plan"}
        
        unsolvable_problem = self._generate_unsolvable_problem(
            solvable_problem, contradiction_plan, contradiction_type)
        
        if not unsolvable_problem:
            return {"status": "failed_to_generate_problem"}
        
        # First verification attempt: autonomous detection
        cot_b, contradiction_detected = self._solve_and_detect_contradiction(unsolvable_problem)
        if not cot_b:
            return {"status": "failed_to_solve"}
        # Second verification attempt: if first failed, use reasoner with explicit contradiction plan
        if not contradiction_detected:
            print("❌ Model did not detect contradiction in CoT_B. Asking deepseek-reasoner to re-evaluate with the expected contradiction point...")
            # Build a focused prompt for the stronger reasoner including our expected contradiction point / plan
            recheck_prompt = (
                "You are a rigorous mathematician and evaluator of reasoning chains. We will provide the full construction chain used to create the candidate unsolvable problem.\n\n"
                "You will receive the following items:\n"
                "1) The ORIGINAL PROBLEM.\n"
                "2) The ORIGINAL MODEL SOLUTION / CHAIN OF THOUGHT (CoT_A) produced when solving the original problem.\n"
                "3) The CONTRADICTION PLAN used to construct the unsolvable variant (including any verification notes produced at generation time).\n"
                "4) The GENERATED UNSOLVABLE PROBLEM text.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "- Carefully trace the construction chain: starting from CoT_A, through the contradiction plan, to the produced unsolvable problem.\n"
                "- Determine whether the construction actually introduces a logical contradiction that makes the problem unsolvable.\n"
                "- If you find ANY contradiction that makes the problem unsolvable, do NOT attempt to produce a solution — instead deliver the EXACT boxed verdict.\n\n"
                "REQUIRED OUTPUT FORMAT (STRICT):\n"
                "[Analysis]\n"
                "- Briefly summarize how the contradiction plan is intended to trigger a conflict given CoT_A (2-6 short bullets).\n"
                "- Point to the specific late-stage step(s) in CoT_A or the generated problem where the conflict arises (quote short snippets).\n\n"
                "[Conclusion]\n"
                "- If contradiction present: OUTPUT EXACTLY \\boxed{Unsolvable} on its own line, then a one-line reason identifying the conflicting conditions and the precise step.\n"
                "- If no contradiction: OUTPUT EXACTLY \\boxed{Solvable} on its own line, then one-line explanation why the plan fails to create a contradiction.\n\n"
                "Do NOT include extraneous commentary outside the two sections above. Be concise.\n\n"
                f"--- ORIGINAL PROBLEM ---\n{solvable_problem}\n\n"
                f"--- ORIGINAL CoT_A ---\n{cot_a}\n\n"
                f"--- CONTRADICTION PLAN (with verification note if present) ---\n{json.dumps(contradiction_plan, ensure_ascii=False, indent=2)}\n\n"
                f"--- GENERATED UNSOLVABLE PROBLEM ---\n{unsolvable_problem}\n\n"
                "Please trace the chain and answer: Does this constructed problem contain the described contradiction and is it truly unsolvable?"
            )
            try:
                # Ask the stronger reasoner for analysis (with one retry allowed)
                recheck_resp = self._call_llm_with_retries(recheck_prompt, temperature=0.0, model_name="deepseek-reasoner", max_attempts=2, backoff_seconds=1.0)
            except Exception as e:
                recheck_resp = None
                print(f"[DEBUG] Reasoner re-check call failed: {e}")

            # Use the same two-step verification: boxed check + LLM verifier
            recheck_detected = False
            if recheck_resp:
                # Step 1: Check boxed conclusion
                keywords = [
                    "contradiction", "unsolvable", "no solution", "no solutions",
                    "conflict", "impossible", "inconsistent", "illogical", "无解", "矛盾", "没有解"
                ]
                boxed_match = re.search(r"\\boxed\{([^}]*)\}", recheck_resp, flags=re.IGNORECASE)
                if boxed_match:
                    boxed_text = boxed_match.group(1).lower()
                    for kw in keywords:
                        if kw in boxed_text:
                            recheck_detected = True
                            print(f"[DEBUG] Reasoner re-check: boxed conclusion indicates unsolvable: {boxed_match.group(1).strip()}")
                            break
                
                # Step 2: If boxed didn't detect, use LLM verifier
                if not recheck_detected:
                    try:
                        verifier = self._verify_with_llm(recheck_resp)
                        v = str(verifier.get("verdict", "UNKNOWN")).upper()
                        if v == "UNSOLVABLE":
                            recheck_detected = True
                            print(f"[DEBUG] Reasoner re-check: LLM verifier accepted as UNSOLVABLE: {verifier}")
                        else:
                            print(f"[DEBUG] Reasoner re-check: LLM verifier result: {verifier}")
                    except Exception as e:
                        print(f"[DEBUG] Reasoner re-check verifier call failed: {e}")

            if recheck_detected:
                contradiction_detected = True
                cot_b = recheck_resp  # Use the recheck response as the final CoT_B
                print("✅ deepseek-reasoner accepted the constructed contradiction on re-check (via boxed+verifier); accepting problem.")
            else:
                print("✖ deepseek-reasoner did not accept the constructed contradiction on re-check.")
                print("➡ Requesting concise change suggestions to improve the unsolvable intent...")
                suggestion_obj = self._suggest_changes_for_unsolvable(solvable_problem, cot_a, contradiction_plan, unsolvable_problem)
                if suggestion_obj:
                    print(f"[INFO] Received change suggestions: {suggestion_obj.get('suggestions')}")
                else:
                    print("[INFO] No concise suggestions returned by reasoner.")
                print("➡ Initiating fallback improvement attempt to strengthen contradiction...")
                improved_problem = self._improve_unsolvable_problem(solvable_problem, cot_a, contradiction_plan, unsolvable_problem)
                if not improved_problem:
                    print("❌ Improvement attempt failed; aborting.")
                    return {"status": "failed_verification", "change_suggestions": suggestion_obj}
                # Re-run autonomous detection on improved problem
                improved_cot_b, improved_detected = self._solve_and_detect_contradiction(improved_problem)
                if not improved_cot_b:
                    print("❌ Improvement solve attempt failed; aborting.")
                    return {"status": "failed_verification_improvement_no_solve", "improved_unsolvable_problem": improved_problem, "change_suggestions": suggestion_obj}
                if not improved_detected:
                    print("❌ Improved problem still not verified as unsolvable.")
                    return {
                        "status": "failed_verification_after_improvement",
                        "improved_unsolvable_problem": improved_problem,
                        "improved_cot_b": improved_cot_b,
                        "change_suggestions": suggestion_obj
                    }
                # Success via improved variant
                print("✅ Fallback improvement produced a verifiable unsolvable problem.")
                unsolvable_problem = improved_problem  # replace for final reporting
                cot_b = improved_cot_b
                contradiction_detected = True
                # Continue to success block
        
        # Success: contradiction detected (either in first or second verification)
        print("\n" + "="*50)
        print("🎯 Construction successful!")
        print(f"Contradiction Type: {contradiction_type}")
        print(f"Construction Strategy: {contradiction_plan.get('strategy', 'Unknown')}")
        print("\n📝 Original Problem Solution (CoT_A):")
        print(to_markdown(cot_a))
        print("\n🔧 Contradiction Construction Plan:")
        print(to_markdown(json.dumps(contradiction_plan, ensure_ascii=False, indent=2)))
        print("\n❓ Generated Unsolvable Problem:")
        print(to_markdown(unsolvable_problem))
        print("\n🔍 Model Solution Attempt (CoT_B):")
        print(to_markdown(cot_b))
        print("="*50)
        
        return {
            "status": "success",
            "contradiction_type": contradiction_type,
            "contradiction_plan": contradiction_plan,
            "unsolvable_problem": unsolvable_problem,
            "cot_a": cot_a,
            "cot_b": cot_b,
            "contradiction_detected": contradiction_detected
        }