import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_affordance_guided import (BLIND_PROMPT, HISTORY_HEADER, OPENROUTER_BASE_URL, POOL_REVIEW,
                                        REVIEW_PROMPT, SCENARIOS, N_DISTRACTORS, ask_judge, pool_test)

import requests
from openai import OpenAI

PROMPT = """Generate an image of a tool that can be used for squeezing juice from a lemon.
            Make sure the tool doesn't resemble existing lemon juicers/squeezers, and that the tool
            can semantically be used to squeeze lemons and all of its parts and components serve a
            purpose. Make the background white and add nothing else, including no lemons."""
OBJECT = "squeeze_juice"
OUT = f"/home/adadt2/nvrd/affordance-test/figures/affordance_guided_novel_objects_v2/{OBJECT}.png"
QUALITY = "high"
MIN_CORRECT = 1  # passes if at least this many of the 3 labelers pick the intended task
MAX_RETRIES = 3  # retries after the first attempt (up to 4 images); each retry uses the judge's rewritten prompt

key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("Set OPENROUTER_API_KEY first (export OPENROUTER_API_KEY=...)")
client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
task, avoid, exclude = SCENARIOS[OBJECT]
pool = [t for t, _, _ in SCENARIOS.values()]

prompt, history = PROMPT, []
for attempt in range(1, MAX_RETRIES + 2):
    r = requests.post("https://openrouter.ai/api/v1/images", timeout=300,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": "openai/gpt-image-2", "prompt": prompt, "quality": QUALITY, "aspect_ratio": "1:1"})
    r.raise_for_status()
    png = base64.b64decode(r.json()["data"][0]["b64_json"])
    with open(OUT, "wb") as f:
        f.write(png)
    b64 = base64.b64encode(png).decode()

    labels = pool_test(client, b64, pool, task, seed=attempt)
    n_ok = sum(l["correct"] for l in labels)
    wrong = sorted({l["choice"] for l in labels if not l["correct"] and l["choice"]})

    blind = ask_judge(client, BLIND_PROMPT, b64)
    hist = (HISTORY_HEADER + "\n".join(
        f"  Attempt {h['attempt']}: blind guesses {h['blind']}; pool picks {h['picks']}; {h['verdict']} -- {h['summary']}"
        for h in history)) if history else ""
    pool_note = POOL_REVIEW.format(n_ok=n_ok, n=len(labels), k=N_DISTRACTORS + 1, min_ok=MIN_CORRECT,
                                   picks=json.dumps([l["choice"] for l in labels]), wrong=", ".join(wrong) or "none")
    review = ask_judge(client, REVIEW_PROMPT.format(guesses=json.dumps(blind.get("guesses", [])), task=task, avoid=avoid,
                                                    exclude=exclude, prompt=prompt, history=hist + pool_note), b64)
    judge_ok = str(review.get("verdict", "FAIL")).upper().startswith("PASS")
    failing = [k for k, v in review.get("criteria", {}).items() if not (isinstance(v, dict) and v.get("ok"))]
    passed = judge_ok and n_ok >= MIN_CORRECT

    print(f"\nattempt {attempt}: {'PASS' if passed else 'FAIL'}  (saved {OUT})")
    print(f"  labelers: {n_ok}/{len(labels)} picked '{task}'")
    for l in labels:
        if not l["correct"]:
            print(f"    {l['model']} picked: {l['choice']}")
    print(f"  judge: {'PASS' if judge_ok else 'FAIL'}" + (f" (failing: {', '.join(failing)})" if failing else "")
          + f" -- {review.get('summary')}")
    print(f"  blind guesses: {blind.get('guesses')}")
    history.append({"attempt": attempt, "blind": blind.get("guesses"), "picks": [l["choice"] for l in labels],
                    "verdict": "pass" if passed else "fail", "summary": review.get("summary")})
    if passed:
        break
    if attempt <= MAX_RETRIES and review.get("revised_prompt"):
        prompt = review["revised_prompt"]
        print(f"  next prompt: {prompt}")

print(f"\nfinal: {'PASS' if passed else 'FAIL'} after {attempt} attempt(s) -> {OUT}")
#with open(os.path.splitext(OUT)[0] + ".json", "w") as f:
    #json.dump({"object": OBJECT, "task": task, "first_prompt": PROMPT, "final_prompt": prompt, "passed": passed, "attempts": history}, f, indent=2)
