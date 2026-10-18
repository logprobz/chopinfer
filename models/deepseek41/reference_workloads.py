"""Reference runner workloads for DeepSeek V4.1 Flash's teacher texts,
long-context texts and image probes.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/deepseek41/reference_workloads.py \\
        --output DIR [--checkpoint CK] [--evals DIR] [--long-tokens 131072]
    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/deepseek41/reference_workloads.py \\
        --answers RUN/f32.json --output DIR [--lock LOCK]
    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/deepseek41/reference_workloads.py --chat --output DIR

Runs on CPU in the runner environment: the checkpoint's tokenizer.json
(through `tokenizers`, which adds no special tokens for this tokenizer) and
its pinned `encoding/encoding.py` (the rendering reference, V4), never the
model. Writes, in DIR (refused if it exists):

- `teacher.json`: the short texts of fixtures/mimo26/teacher-texts.json
  (not its chat and long-context texts), DeepSeek chat texts rendered by
  `encode_messages` (`CHATS`, chat and thinking modes, reasoning efforts and
  a tool call) and `long_4k`, each scored from its second token (`long_4k`
  over its last 512). Plain texts start with the BOS token, as every
  rendered conversation does. Every sequence holds at most `F64_TOKENS`,
  so the F64-TQ run covers the whole teacher set.
- `long-context.json`: `long_4k` first, so the runner's repeat check of each
  mode's first sequence is cheap, then `long_<n>k` of `--long-tokens`
  tokens (default 131,072), scored over its last 1,024. Long texts are
  cleaned wikitext-103 test paragraphs from the pinned evals cache
  (`<evals>/wikitext.json`, SHA256 in its manifest) from a seeded start; the
  scored tail quotes a passage from the first twentieth of the text, so
  those positions can copy across the whole context.
- `probes-greedy.json` and one `<probe>.bf16` / `<probe>-grid.i64` pair per
  probe: the PNGs of probe_images.py in a user turn with `PROBE_QUESTION`
  (chat mode), each image turned into the vendor's BF16 patches and
  [n_vit_h, n_vit_w] grid by the pinned `inference/image_processor.py`
  (`load_image`) and its span expanded as `prepare_vl_inputs` does; a
  free-text probe with a greedy answer of at most `MAX_NEW_TOKENS` ending at
  the end-of-sentence token.
- `capture/`: the probes' names, images, question and prompt token IDs, the
  PNG digests and a manifest, for a media gate.
- `manifest.json`: the digests of every input (tokenizer, encoding.py,
  image processor and config, wikitext, each probe PNG and its patches) and
  each sequence's length and token-ID SHA256.
- `chat.json` (`--chat` alone, with its own manifest): the chat
  reference's texts, CHATS and CHATS_EXTENDED rendered by `encode_messages`
  and scored from their second token, the set the `chat` check gates.

With `--answers RUN/f32.json`, DIR already holds those files and gains
`probes.json`: the probes forcing the F32 run's greedy answers (checked to
be this workload's sequences, model and checkpoint lock).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/qual"))
sys.path.insert(0, str(ROOT / "models/deepseek41"))

SCHEMA = "chopinfer.refrun.workload.v1"
MODEL = "deepseek41"
SEED = 20261017
F64_TOKENS = 4096
LONG_4K, LONG_4K_SCORED = 4096, 512
LONG_SCORED = 1024
MIN_LONG_TOKENS = 65536
ENCODING = "encoding/encoding.py"
ENCODING_SHA256 = "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2"
BOS = "<｜begin▁of▁sentence｜>"
IMAGE_PROCESSOR = "inference/image_processor.py"
IMAGE_PROCESSOR_SHA256 = "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272"
CONFIG = "inference/config.json"
LOCK = ROOT / "models/deepseek41/checkpoint-files.json"
PROBE_QUESTION = "Describe this image in one short paragraph."
MAX_NEW_TOKENS = 256
# <｜end▁of▁sentence｜>, the config's eos_token_id.
STOP_IDS = (1,)
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}
# DeepSeek chat texts, the counterparts of GLM's, MiMo's and Qwen's chat
# teacher texts: each conversation ends with an assistant turn.
CHATS = {
    "chat_system": {"thinking_mode": "chat", "messages": [
        {"role": "system", "content": "You are a concise assistant. Answer in at most two sentences."},
        {"role": "user", "content": "Why is the sky blue?"},
        {"role": "assistant",
         "content": "Sunlight is scattered by air molecules, and shorter blue wavelengths scatter much more "
                    "strongly than red ones. That scattered blue light reaches our eyes from every direction."}]},
    "chat_thinking": {"thinking_mode": "thinking", "messages": [
        {"role": "user", "content": "Is 91 a prime number?"},
        {"role": "assistant", "reasoning_content":
            "I need to check small divisors. 91 is odd, so 2 does not divide it. The digit sum is 10, so 3 "
            "does not divide it. It does not end in 0 or 5. 7 * 13 = 91, so 91 is composite.",
         "content": "No. 91 = 7 * 13, so it is not prime."}]},
    "chat_effort_low": {"thinking_mode": "thinking", "reasoning_effort": "low", "messages": [
        {"role": "user", "content": "Name three prime numbers greater than 50."},
        {"role": "assistant", "reasoning_content": "53, 59 and 61 are prime.", "content": "53, 59 and 61."}]},
    "chat_effort_max": {"thinking_mode": "thinking", "reasoning_effort": "max", "messages": [
        {"role": "user", "content": "What is 17 squared?"},
        {"role": "assistant", "reasoning_content": "17 * 17 = 289.", "content": "289."}]},
    "chat_multi_turn": {"thinking_mode": "chat", "messages": [
        {"role": "user", "content": "Give me a name for a gray cat."},
        {"role": "assistant", "content": "How about Pewter? It is a soft gray metal, which suits a gray cat."},
        {"role": "user", "content": "Something shorter, please."},
        {"role": "assistant", "content": "Try Ash, Smoke or Slate."}]},
    "chat_zh": {"thinking_mode": "chat", "messages": [
        {"role": "user", "content": "请用一句话解释什么是机器学习。"},
        {"role": "assistant",
         "content": "机器学习是让计算机从数据中自动发现规律，并利用这些规律对新数据做出预测或决策的方法。"}]},
    "chat_tool_call": {"thinking_mode": "thinking", "messages": [
        {"role": "system", "content": "You can call tools.", "tools": [WEATHER]},
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "reasoning_content": "The user wants the current weather, so call get_weather.",
         "content": "", "tool_calls": [{"type": "function", "function": {
             "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "content": "{\"temperature\": 18, \"condition\": \"cloudy\"}"},
        {"role": "assistant", "reasoning_content": "The tool says 18 degrees and cloudy.",
         "content": "It is 18 degrees Celsius and cloudy in Paris."}]},
}


def tool_call(name, arguments):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


FORECAST = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather forecast for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "days": {"type": "integer", "description": "Number of forecast days, 1 to 7"},
        "metric": {"type": "boolean", "description": "Celsius when true, Fahrenheit when false"}},
        "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search_papers", "description": "Search a library catalog of research papers.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"},
        "filters": {"type": "object", "properties": {
            "year_from": {"type": "integer"}, "venues": {"type": "array", "items": {"type": "string"}}}},
        "limit": {"type": "integer"}},
        "required": ["query"]}}}
CALENDAR = {"type": "function", "function": {
    "name": "create_event", "description": "Create a calendar event.",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string"}, "start": {"type": "string", "description": "ISO 8601 date and time"},
        "duration_minutes": {"type": "integer"}, "attendees": {"type": "array", "items": {"type": "string"}}},
        "required": ["title", "start"]}}}


# The extended chat set, the same conversations as GLM's (models/glm5next/
# reference_workloads.py) in encode_messages' form: tools on the system
# message (one is added where the conversation has none), thinking mode
# where an assistant turn reasons, chat mode otherwise. With CHATS they form
# the `chat` reference (`--chat`); the teacher set keeps CHATS alone.
CHATS_EXTENDED = {
    "chat_tool_two_cities": {"messages": [
        {"role": "system", "content": "You are a travel assistant. Use the tools for live data and keep answers short."},
        {"role": "user", "content": "I fly from Lisbon to Oslo on Friday. What weather should I pack for in both "
                                    "cities over the next three days?"},
        {"role": "assistant", "reasoning_content":
            "The user needs a three-day forecast for Lisbon and Oslo. The two calls are independent, so I can "
            "make both at once.",
         "content": "", "tool_calls": [
             tool_call("get_weather", {"city": "Lisbon", "days": 3, "metric": True}),
             tool_call("get_weather", {"city": "Oslo", "days": 3, "metric": True})]},
        {"role": "tool", "content": "{\"city\": \"Lisbon\", \"forecast\": [{\"day\": \"Fri\", \"high\": 24, \"low\": 16, "
                                    "\"sky\": \"sunny\"}, {\"day\": \"Sat\", \"high\": 23, \"low\": 15, \"sky\": \"sunny\"}, "
                                    "{\"day\": \"Sun\", \"high\": 21, \"low\": 15, \"sky\": \"cloudy\"}]}"},
        {"role": "tool", "content": "{\"city\": \"Oslo\", \"forecast\": [{\"day\": \"Fri\", \"high\": 9, \"low\": 2, "
                                    "\"sky\": \"rain\"}, {\"day\": \"Sat\", \"high\": 7, \"low\": 1, \"sky\": \"rain\"}, "
                                    "{\"day\": \"Sun\", \"high\": 8, \"low\": 0, \"sky\": \"cloudy\"}]}"},
        {"role": "assistant", "reasoning_content":
            "Lisbon is warm and dry, 21 to 24 by day. Oslo is cold and wet, highs 7 to 9 and lows near freezing.",
         "content": "Lisbon will be warm and mostly sunny (highs of 21 to 24 °C), so light clothes are enough there. "
                    "Oslo will be cold and rainy, with highs of 7 to 9 °C and nights near freezing: pack a warm "
                    "layer, a waterproof jacket and shoes that can take the rain."}],
        "tools": [FORECAST]},
    "chat_tool_nested_args": {"messages": [
        {"role": "user", "content": "Find me three recent papers on speculative decoding for large language models, "
                                    "from 2023 or later, published at NeurIPS or ICML."},
        {"role": "assistant", "reasoning_content":
            "I should search the catalog with a year filter and the two venues, limited to three results.",
         "content": "", "tool_calls": [tool_call("search_papers", {
             "query": "speculative decoding large language models",
             "filters": {"year_from": 2023, "venues": ["NeurIPS", "ICML"]}, "limit": 3})]},
        {"role": "tool", "content": "[{\"title\": \"Fast Inference from Transformers via Speculative Decoding\", "
                                    "\"venue\": \"ICML\", \"year\": 2023}, {\"title\": \"Medusa: Simple LLM Inference "
                                    "Acceleration Framework with Multiple Decoding Heads\", \"venue\": \"ICML\", "
                                    "\"year\": 2024}, {\"title\": \"EAGLE: Speculative Sampling Requires Rethinking "
                                    "Feature Uncertainty\", \"venue\": \"ICML\", \"year\": 2024}]"},
        {"role": "assistant", "reasoning_content":
            "The search returned three ICML papers that match. I will list them with one line each on the idea.",
         "content": "Here are three that match:\n\n1. **Fast Inference from Transformers via Speculative Decoding** "
                    "(ICML 2023): a small draft model proposes tokens and the large model verifies them in one pass, "
                    "with output identical to sampling from the large model.\n2. **Medusa** (ICML 2024): extra "
                    "decoding heads on the base model propose several future tokens, verified with tree attention.\n"
                    "3. **EAGLE** (ICML 2024): drafting at the feature level, one layer above the output, which "
                    "raises the acceptance rate.\n\nNone of the results were from NeurIPS; I can widen the search "
                    "if you want more."}],
        "tools": [SEARCH]},
    "chat_tool_retry": {"messages": [
        {"role": "user", "content": "Put a 45 minute design review with Ana and Wei on my calendar for next Tuesday at 3pm."},
        {"role": "assistant", "reasoning_content":
            "Next Tuesday is 2026-03-17. 3pm is 15:00. I will create the event with both attendees.",
         "content": "", "tool_calls": [tool_call("create_event", {
             "title": "Design review", "start": "2026-03-17 15:00", "duration_minutes": 45,
             "attendees": ["Ana", "Wei"]})]},
        {"role": "tool", "content": "{\"error\": \"invalid start: expected ISO 8601, for example 2026-03-17T15:00:00\"}"},
        {"role": "assistant", "reasoning_content":
            "The start time must use the ISO 8601 form with a T separator and seconds. I will retry with that.",
         "content": "", "tool_calls": [tool_call("create_event", {
             "title": "Design review", "start": "2026-03-17T15:00:00", "duration_minutes": 45,
             "attendees": ["Ana", "Wei"]})]},
        {"role": "tool", "content": "{\"id\": \"evt_8812\", \"status\": \"created\"}"},
        {"role": "assistant", "reasoning_content": "The event was created.",
         "content": "Done: \"Design review\" is on your calendar for Tuesday, March 17 at 15:00 for 45 minutes, "
                    "with Ana and Wei invited."}],
        "tools": [CALENDAR], "kwargs": {"reasoning_effort": "low"}},
    "chat_code_fix": {"messages": [
        {"role": "user", "content": "This function should return the median of a list, but it gives wrong answers "
                                    "for some inputs. What is wrong?\n\n```python\ndef median(values):\n"
                                    "    values.sort()\n    middle = len(values) / 2\n    if len(values) % 2:\n"
                                    "        return values[middle]\n    return (values[middle] + values[middle + 1]) / 2\n```"},
        {"role": "assistant", "reasoning_content":
            "Three problems. First, len(values) / 2 is a float in Python 3, so indexing fails with a TypeError. "
            "Second, for an even length n the two middle elements are at n // 2 - 1 and n // 2, not n // 2 and "
            "n // 2 + 1; for [1, 2, 3, 4] the code would average 3 and 4 instead of 2 and 3. Third, sort() "
            "changes the caller's list. Use sorted() and integer division.",
         "content": "There are three bugs:\n\n1. `len(values) / 2` is a float in Python 3, so `values[middle]` "
                    "raises a `TypeError`. Use `//`.\n2. For an even length the two middle elements are at "
                    "`n // 2 - 1` and `n // 2`. The code averages the element after the middle instead, so "
                    "`[1, 2, 3, 4]` would give 3.5 rather than 2.5.\n3. `values.sort()` reorders the caller's "
                    "list as a side effect.\n\nA corrected version:\n\n```python\ndef median(values):\n"
                    "    ordered = sorted(values)\n    n = len(ordered)\n    middle = n // 2\n    if n % 2:\n"
                    "        return ordered[middle]\n    return (ordered[middle - 1] + ordered[middle]) / 2\n```\n\n"
                    "It still raises an `IndexError` on an empty list, which is a reasonable signal that the "
                    "median is undefined there."}],
        "kwargs": {"reasoning_effort": "high"}},
    "chat_math_reasoning": {"messages": [
        {"role": "user", "content": "A bakery sells muffins at $3 each or a box of 6 for $15. Priya needs 20 muffins "
                                    "for a party. What is the least she can pay?"},
        {"role": "assistant", "reasoning_content":
            "Boxes cost 15 / 6 = 2.50 per muffin, cheaper than 3. So buy as many boxes as make sense. "
            "Three boxes give 18 muffins for 45, plus 2 single muffins for 6: 51 total. Four boxes give 24 muffins "
            "for 60, which is more. Two boxes give 12 for 30 plus 8 singles for 24: 54. So three boxes and two "
            "singles is cheapest, at $51. Check: 18 + 2 = 20 muffins.",
         "content": "Boxes work out to $2.50 per muffin, cheaper than buying singly, so she should buy as many "
                    "full boxes as she can use:\n\n- 3 boxes = 18 muffins for $45\n- 2 single muffins for $6\n\n"
                    "That is 20 muffins for **$51**. A fourth box would cost $60, and only two boxes plus eight "
                    "singles would cost $54."}],
        "kwargs": {"reasoning_effort": "high"}},
    "chat_multi_turn_plan": {"messages": [
        {"role": "user", "content": "I want to start running. I can do about 10 minutes now without stopping. "
                                    "Can you suggest a plan to get to 30 minutes?"},
        {"role": "assistant", "reasoning_content": "",
         "content": "A gradual plan over about six weeks works well. Run three days a week with a rest day between "
                    "runs.\n\n- Weeks 1-2: 12 to 15 minutes of easy running, walking briefly if you need to.\n"
                    "- Weeks 3-4: 18 to 22 minutes, keeping the pace conversational.\n- Weeks 5-6: 25 to 30 "
                    "minutes.\n\nIncrease the time by no more than about 10% per week, and repeat a week if it "
                    "felt hard."},
        {"role": "user", "content": "My knees ache a bit after running. Should I stop?"},
        {"role": "assistant", "reasoning_content": "",
         "content": "Mild aching that fades within a day is common when you start. Sharp pain, swelling, or pain "
                    "that gets worse while you run is a reason to stop and see a doctor or physiotherapist.\n\n"
                    "To reduce the strain: run on softer surfaces such as trails or a track, check that your shoes "
                    "are not worn out, shorten your stride slightly, and add two short strength sessions a week "
                    "for your hips and thighs."},
        {"role": "user", "content": "Thanks. How do I know if I'm going too fast?"},
        {"role": "assistant", "reasoning_content": "",
         "content": "Use the talk test: at an easy pace you can speak in full sentences. If you can only get out a "
                    "few words at a time, slow down. Most of your running should feel easy enough that you could "
                    "keep going a little longer when you finish."}]},
    "chat_system_format": {"messages": [
        {"role": "system", "content": "You are a support assistant for a home internet provider. Rules: answer in "
                                      "at most four numbered steps; never ask for the customer's password; if the "
                                      "problem is an outage, give the status page address status.example.net; end "
                                      "every answer with the line \"Reply 'agent' to reach a person.\""},
        {"role": "user", "content": "My wifi works on my phone but my laptop says connected, no internet. "
                                    "I already restarted the laptop."},
        {"role": "assistant", "reasoning_content":
            "The phone works, so the line is up and this is not an outage. The laptop has a link but no internet, "
            "which points to its network settings: renew the address, forget and rejoin the network, check DNS, "
            "or disable a VPN.",
         "content": "1. Turn wifi off and on again on the laptop, then forget the network and rejoin it.\n2. If it "
                    "still fails, open a terminal and renew the address (on Windows, `ipconfig /release` and then "
                    "`ipconfig /renew`).\n3. Turn off any VPN or proxy and set DNS back to automatic.\n4. Try a "
                    "website by its address, for example 1.1.1.1; if that loads, the problem is DNS.\n\n"
                    "Reply 'agent' to reach a person."}],
        "kwargs": {"reasoning_effort": "low"}},
    "chat_zh_explain": {"messages": [
        {"role": "system", "content": "你是一位耐心的中学物理老师。"},
        {"role": "user", "content": "为什么夏天柏油路面上方的空气看起来会抖动？"},
        {"role": "assistant", "reasoning_content":
            "这是光的折射。路面被晒热，贴近路面的空气温度高、密度小，折射率比上方的冷空气低。热空气不断上升并和冷空气混合，"
            "折射率分布一直在变化，所以穿过这层空气的光线路径不断改变，看到的景物就在抖动。",
         "content": "这是光的折射造成的。\n\n夏天路面被太阳晒得很热，贴近路面的空气也被加热，密度变小，折射率比上方较冷的"
                    "空气低。光从冷空气进入热空气时会发生偏折。\n\n热空气还在不断上升，和冷空气混在一起，所以各处的折射率"
                    "时刻在变化，光线的路径也跟着不停改变。我们透过这层空气看远处的景物，它的位置就会轻微晃动，看起来像"
                    "在抖动。"},
        {"role": "user", "content": "那远处路面上像水一样的反光也是同一个原因吗？"},
        {"role": "assistant", "reasoning_content":
            "是的，这是下蜃景。来自天空的光斜着射向路面，经过越来越热的空气层时不断向上弯曲，最后进入眼睛，"
            "看起来像是从路面反射来的，所以像一滩水。",
         "content": "是同一个原理，叫作下蜃景。天空的光斜着射向路面时，穿过越来越热的空气层，会一点点向上弯曲，"
                    "最后进入你的眼睛。大脑以为光是沿直线传来的，就把天空的像看成在路面上，好像那里有一滩反光的水。"
                    "走近之后角度变了，“水”也就不见了。"}]},
    "chat_json_extract": {"messages": [
        {"role": "user", "content": "Extract the orders from this message as a JSON list with fields customer, item, "
                                    "quantity and deliver_by (YYYY-MM-DD, or null if not given).\n\n\"Hi, this is "
                                    "Marta from Café Lumen. Please send 12 bags of the Ethiopian beans and 4 bottles of "
                                    "oat milk by March 3rd, 2026. Also, my colleague Jonas at the Riverside shop needs "
                                    "two grinders, no rush on those.\""},
        {"role": "assistant", "reasoning_content":
            "Two customers. Marta (Café Lumen): 12 bags Ethiopian beans and 4 bottles oat milk, both by 2026-03-03. "
            "Jonas (Riverside shop): 2 grinders, no date, so null. Quantities as integers.",
         "content": "```json\n[\n  {\"customer\": \"Marta (Café Lumen)\", \"item\": \"Ethiopian beans\", "
                    "\"quantity\": 12, \"deliver_by\": \"2026-03-03\"},\n  {\"customer\": \"Marta (Café Lumen)\", "
                    "\"item\": \"oat milk\", \"quantity\": 4, \"deliver_by\": \"2026-03-03\"},\n  {\"customer\": "
                    "\"Jonas (Riverside shop)\", \"item\": \"grinder\", \"quantity\": 2, \"deliver_by\": null}\n]\n```"}]},
    "chat_summary": {"messages": [
        {"role": "user", "content": "Summarize this in two sentences for a newsletter:\n\nThe city council voted on "
                                    "Tuesday to convert the old freight yard north of the river into a public park. The "
                                    "site, about 14 hectares, has been empty since the rail company moved its operations "
                                    "in 2019. Under the plan, the first section, with playgrounds, a community garden and "
                                    "a cycling path along the water, would open in the spring of 2028. Two historic "
                                    "warehouses will be kept and turned into a market hall and studio space. The project "
                                    "is expected to cost 38 million euros, of which a regional fund covers about a third. "
                                    "Some residents asked for more housing on the site instead, and the council agreed to "
                                    "study a small residential strip along the eastern edge."},
        {"role": "assistant", "reasoning_content":
            "Key points: council approved turning the 14-hectare freight yard into a park; first section opens spring "
            "2028 with playgrounds, garden, cycle path; warehouses become market hall and studios; 38 million euros, a "
            "third from a regional fund; housing strip to be studied.",
         "content": "The city council has approved turning the 14-hectare former freight yard north of the river "
                    "into a public park, with playgrounds, a community garden and a riverside cycling path opening in "
                    "spring 2028 and two historic warehouses becoming a market hall and studios. The project will "
                    "cost 38 million euros, about a third of it from a regional fund, and the council will also "
                    "study a small strip of housing along the site's eastern edge."}],
        "kwargs": {"reasoning_effort": "low"}},
}


def encode_spec(spec):
    """A CHATS_EXTENDED spec ({"messages", "tools", "kwargs"}) as a CHATS one:
    thinking mode when an assistant turn reasons, else chat mode without the
    empty reasoning; the tools on the system message (a short one is added
    where the conversation has none); the reasoning effort from kwargs."""
    messages = [dict(m) for m in spec["messages"]]
    thinking = any(m.get("reasoning_content") for m in messages)
    if not thinking:
        for m in messages:
            m.pop("reasoning_content", None)
    if spec.get("tools"):
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": "You can call tools."})
        messages[0]["tools"] = spec["tools"]
    result = {"thinking_mode": "thinking" if thinking else "chat", "messages": messages}
    if "reasoning_effort" in spec.get("kwargs", {}):
        result["reasoning_effort"] = spec["kwargs"]["reasoning_effort"]
    return result


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def ids_sha256(ids):
    return sha256(json.dumps(ids).encode())


def tokenizer(checkpoint):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))


def encoding(checkpoint):
    """The checkpoint's encoding.py, executed from its pinned bytes."""
    path = Path(checkpoint) / ENCODING
    data = path.read_bytes()
    if sha256(data) != ENCODING_SHA256:
        raise ValueError(f"{ENCODING} is not the pinned {ENCODING_SHA256}")
    module = types.ModuleType("deepseek41_encoding")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def chat_rows(renderer, chats):
    """The conversations rendered by encode_messages (BOS included)."""
    return [{"name": name, "category": "chat",
             "text": renderer.encode_messages(spec["messages"], thinking_mode=spec["thinking_mode"],
                                              reasoning_effort=spec.get("reasoning_effort"))}
            for name, spec in chats.items()]


def scored(tok, rows):
    return [{"name": row["name"], "category": row["category"], "token_ids": encode(tok, row["text"]),
             "score_from": 1} for row in rows]


def chat_texts(tok, renderer):
    """The chat reference's texts: CHATS, then CHATS_EXTENDED, each scored from its second token."""
    extended = {name: encode_spec(spec) for name, spec in CHATS_EXTENDED.items()}
    return scored(tok, chat_rows(renderer, CHATS | extended))


def short_texts(tok, renderer):
    """The model-neutral short teacher texts (after BOS) and the DeepSeek chat texts, tokenized."""
    source = json.loads((ROOT / "fixtures/mimo26/teacher-texts.json").read_text())
    rows = [{"name": t["name"], "category": t["category"], "text": BOS + t["text"]} for t in source["texts"]
            if t["category"] not in ("chat", "long-context")]
    return scored(tok, rows + chat_rows(renderer, CHATS))


def corpus(evals):
    """Cleaned wikitext-103 test paragraphs of the pinned evals cache."""
    from evals import paragraphs
    manifest = json.loads((Path(evals) / "manifest.json").read_text())["wikitext"]
    raw = (Path(evals) / "wikitext.json").read_bytes()
    if sha256(raw) != manifest["sha256"]:
        raise ValueError("wikitext.json does not match its manifest")
    return paragraphs(json.loads(raw)), {"revision": manifest["revision"], "sha256": manifest["sha256"]}


def long_text(tok, paragraphs, length, scored_tokens):
    """`length` tokens: BOS, wikitext from a seeded start, and the last
    `scored_tokens` quoting a passage that starts at a paragraph break after
    5% of the text."""
    rng = random.Random(f"{SEED}:deepseek41-long")
    start = rng.randrange(len(paragraphs))
    ordered = paragraphs[start:] + paragraphs[:start]
    bos = encode(tok, BOS)
    intro = encode(tok, "\n\nThe following passage appeared earlier in this text:\n\n")
    body = []
    while len(body) < length:
        for paragraph in ordered:
            body.extend(encode(tok, paragraph + "\n\n"))
            if len(body) >= length:
                break
    body = body[:length - scored_tokens - len(intro) - len(bos)]
    first = next(i for i in range(len(body) // 20, len(body)) if tok.decode(body[i - 1:i]).endswith("\n"))
    prompt = bos + body + intro + body[first:first + scored_tokens]
    if len(prompt) != length:
        raise ValueError(f"the long text has {len(prompt)} tokens, not {length}")
    return {"name": f"long_{length // 1024}k", "category": "long-context", "token_ids": prompt,
            "score_from": length - scored_tokens, "quote_source_start": len(bos) + first}


def image_processor(checkpoint):
    """The checkpoint's image_processor.py, executed from its pinned bytes."""
    path = Path(checkpoint) / IMAGE_PROCESSOR
    data = path.read_bytes()
    if sha256(data) != IMAGE_PROCESSOR_SHA256:
        raise ValueError(f"{IMAGE_PROCESSOR} is not the pinned {IMAGE_PROCESSOR_SHA256}")
    module = types.ModuleType("deepseek41_image_processor")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def processor_identity(checkpoint):
    return {"code_sha256": IMAGE_PROCESSOR_SHA256,
            "config_sha256": sha256((Path(checkpoint) / CONFIG).read_bytes())}


def probe_prompt(tok, renderer, processor, args, png):
    """(token IDs with the image span expanded, BF16 patches, [n_vit_h, n_vit_w]):
    encode_messages renders the turn with one image placeholder, load_image
    makes the patches, and the placeholder becomes the vendor's span
    (image_processor.prepare_vl_inputs)."""
    messages = [{"role": "user", "content": [{"type": "image", "data": png},
                                             {"type": "text", "text": PROBE_QUESTION}]}]
    prompt, media = renderer.encode_messages(messages, thinking_mode="chat", return_multi_modal_data=True)
    if len(media["images"]) != 1:
        raise ValueError("a probe holds one image")
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = processor.load_image(media["images"][0], args)
    span = processor.image_token_types(n_llm_h, n_llm_w).numel()
    ids = []
    for token in encode(tok, prompt):
        ids.extend([args.image_token_id] * span if token == args.image_token_id else [token])
    if ids.count(args.image_token_id) != span:
        raise ValueError("the rendered probe does not hold exactly one image placeholder")
    return ids, patches, (n_vit_h, n_vit_w)


def probes(tok, renderer, checkpoint, output):
    """The image probes with greedy answers, and the request capture (`capture/`)."""
    import probe_images
    import torch
    processor = image_processor(checkpoint)
    args = types.SimpleNamespace(**json.loads((Path(checkpoint) / CONFIG).read_text()))
    sequences, digests, capture = [], {}, []
    pngs = probe_images.pngs()
    for name, data in pngs.items():
        ids, patches, grid = probe_prompt(tok, renderer, processor, args, data)
        raw = patches.contiguous().view(torch.int16).numpy().tobytes()
        (output / f"{name}.bf16").write_bytes(raw)
        (output / f"{name}-grid.i64").write_bytes(torch.tensor(grid, dtype=torch.int64).numpy().tobytes())
        digests[name] = {"png_sha256": sha256(data), "grid": list(grid), "patches_sha256": sha256(raw)}
        media = {"kind": "image", "tensor": {"path": f"{name}.bf16", "dtype": "bf16", "shape": list(patches.shape)},
                 "grid": {"path": f"{name}-grid.i64", "dtype": "i64", "shape": [2]},
                 "processor": processor_identity(checkpoint)}
        sequences.append({"name": name, "category": "free-text", "token_ids": ids, "score_from": len(ids),
                          "media": [media],
                          "answer": {"policy": "greedy", "max_new_tokens": MAX_NEW_TOKENS, "stop_ids": list(STOP_IDS)}})
        capture.append({"name": name, "images": [name], "text": PROBE_QUESTION, "token_ids": ids})
    folder = output / "capture"
    folder.mkdir()
    data = (json.dumps(capture) + "\n").encode()
    (folder / "probes.json").write_bytes(data)
    (folder / "identity.json").write_text(json.dumps({"images": {n: sha256(b) for n, b in pngs.items()}}) + "\n")
    (folder / "manifest.json").write_text(json.dumps({"files": {"probes.json": {"sha256": sha256(data)}}}) + "\n")
    return sequences, digests


def answered(output, run, lock=LOCK):
    """`probes-greedy.json` with each probe forcing the F32 run's greedy
    answer, checked to be this workload's sequence, model and lock."""
    from tools.model_sdk.diagnostic import canonical
    from tools.refrun import keys, workload
    model, records = workload.load(output / "probes-greedy.json")
    document = json.loads(Path(run).read_text())
    if document["policy"]["name"] != "f32":
        raise ValueError(f"{run} is not an f32 run")
    source_lock = sha256(canonical(keys.read_lock(lock)))
    entries = {e["name"]: e for e in document["entries"] if e["key"]["sequence"]["answer"]["policy"] == "greedy"}
    if sorted(entries) != sorted(r.name for r in records):
        raise ValueError(f"{run} does not answer every probe of probes-greedy.json")
    sequences = []
    for record, spec in zip(records, json.loads((output / "probes-greedy.json").read_text())["sequences"]):
        key = entries[record.name]["key"]
        if model != MODEL or key["model"] != MODEL:
            raise ValueError(f"{record.name}: the run is of model {key['model']}, not {MODEL}")
        if key["checkpoint"]["source_lock_sha256"] != source_lock:
            raise ValueError(f"{record.name}: the run read another checkpoint lock")
        if key["sequence"] != keys.sequence(record):
            raise ValueError(f"{record.name}: the run scored another sequence")
        sequences.append(spec | {"answer": {"policy": "forced", "ids": entries[record.name]["scores"]["answer"]}})
    for sequence in sequences:
        workload.sequence(sequence, output)
    return sequences


def write(path, sequences):
    with path.open("x") as stream:
        stream.write(json.dumps({"schema": SCHEMA, "model": MODEL, "sequences": sequences}) + "\n")


def build(checkpoint, evals, output, long_tokens):
    if long_tokens < MIN_LONG_TOKENS or long_tokens % 1024:
        raise ValueError(f"--long-tokens must be a multiple of 1024 of at least {MIN_LONG_TOKENS}")
    checkpoint = Path(checkpoint)
    tok, renderer = tokenizer(checkpoint), encoding(checkpoint)
    paragraphs, wikitext = corpus(evals)
    long_4k = long_text(tok, paragraphs, LONG_4K, LONG_4K_SCORED)
    long = long_text(tok, paragraphs, long_tokens, LONG_SCORED)
    teacher = short_texts(tok, renderer) + [long_4k]
    for sequence in teacher:
        if len(sequence["token_ids"]) > F64_TOKENS:
            raise ValueError(f"{sequence['name']}: {len(sequence['token_ids'])} tokens exceed the f64 limit")
    output.mkdir(parents=True)

    def plain(sequence):
        return {k: v for k, v in sequence.items() if k != "quote_source_start"}
    media, images = probes(tok, renderer, checkpoint, output)
    write(output / "teacher.json", [plain(s) for s in teacher])
    write(output / "long-context.json", [plain(long_4k), plain(long)])
    write(output / "probes-greedy.json", media)
    manifest = {
        "schema": "chopinfer.deepseek41.reference-workloads.v1",
        "generator": "models/deepseek41/reference_workloads.py",
        "tokenizer_json_sha256": sha256((checkpoint / "tokenizer.json").read_bytes()),
        "encoding_py_sha256": ENCODING_SHA256, "image_processor": processor_identity(checkpoint),
        "wikitext": wikitext, "images": images,
        "long_quote_source_start": {s["name"]: s["quote_source_start"] for s in (long_4k, long)},
        "sequences": {kind: [{"name": s["name"], "tokens": len(s["token_ids"]), "score_from": s["score_from"],
                              "token_ids_sha256": ids_sha256(s["token_ids"])} for s in sequences]
                      for kind, sequences in (("teacher", teacher), ("long-context", [long_4k, long]),
                                              ("probes", media))}}
    with (output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    return manifest


def build_chat(checkpoint, output):
    """The chat reference's workload in `output` (refused if it exists):
    `chat.json` and `manifest.json` with the tokenizer and encoding.py
    digests and each sequence's length and token-ID SHA256."""
    checkpoint = Path(checkpoint)
    sequences = chat_texts(tokenizer(checkpoint), encoding(checkpoint))
    for sequence in sequences:
        if len(sequence["token_ids"]) > F64_TOKENS:
            raise ValueError(f"{sequence['name']}: {len(sequence['token_ids'])} tokens exceed the f64 limit")
    output.mkdir(parents=True)
    write(output / "chat.json", sequences)
    manifest = {
        "schema": "chopinfer.deepseek41.reference-workloads.v1",
        "generator": "models/deepseek41/reference_workloads.py",
        "tokenizer_json_sha256": sha256((checkpoint / "tokenizer.json").read_bytes()),
        "encoding_py_sha256": ENCODING_SHA256,
        "sequences": {"chat": [{"name": s["name"], "tokens": len(s["token_ids"]), "score_from": s["score_from"],
                                "token_ids_sha256": ids_sha256(s["token_ids"])} for s in sequences]}}
    with (output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", default=os.environ.get("CHOPINFER_DEEPSEEK_CHECKPOINT"),
                        help="checkpoint directory (tokenizer.json, encoding/encoding.py)")
    parser.add_argument("--evals", type=Path, default=Path(os.environ.get("CHOPINFER_DATA", "")) / "evals",
                        help="the evals cache holding wikitext.json and manifest.json")
    parser.add_argument("--long-tokens", type=int, default=131072)
    parser.add_argument("--answers", type=Path, help="the runner's f32.json of probes-greedy.json; "
                                                     "writes probes.json into the existing --output")
    parser.add_argument("--lock", type=Path, default=LOCK, help="the checkpoint lock the run read")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chat", action="store_true",
                        help="write only the chat reference's workload: chat.json and its manifest")
    args = parser.parse_args()
    if args.chat:
        if not args.checkpoint:
            parser.error("set CHOPINFER_DEEPSEEK_CHECKPOINT or --checkpoint")
        manifest = build_chat(args.checkpoint, args.output)
        print(json.dumps([(s["name"], s["tokens"]) for s in manifest["sequences"]["chat"]]))
        return
    if args.answers:
        write(args.output / "probes.json", answered(args.output, args.answers, args.lock))
        return
    if not args.checkpoint:
        parser.error("set CHOPINFER_DEEPSEEK_CHECKPOINT or --checkpoint")
    manifest = build(args.checkpoint, args.evals, args.output, args.long_tokens)
    print(json.dumps({kind: [(s["name"], s["tokens"]) for s in rows]
                      for kind, rows in manifest["sequences"].items()}))


if __name__ == "__main__":
    main()
