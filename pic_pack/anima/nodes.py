"""
Anima Prompt Forge: PMI 共现扩充 + Prompt 组装
基于 Danbooru 共现统计（PMI）做标签扩充，不依赖向量检索。
配合本地 GGUF LLM 节点使用。
"""

import gc
import sys
import json
import random
import re
from pathlib import Path

import torch
import comfy.model_management as model_management

# ── 加载共现 PMI 表（替代 RAG 语义检索）──────────────────
# 数据由 scripts/build_cooccurrence_pmi.py 预生成。
# 格式: {tag: [[other_tag, score], ...top30]}
_PMI_TABLE = None
_PMI_PATH = Path(__file__).resolve().parent / "data" / "cooccurrence_pmi.json"


def _load_pmi_table():
    """懒加载共现 PMI 表，缓存到全局"""
    global _PMI_TABLE
    if _PMI_TABLE is not None:
        return _PMI_TABLE
    if _PMI_PATH.exists():
        try:
            _PMI_TABLE = json.loads(_PMI_PATH.read_text(encoding="utf-8"))
            print(f"[Anima-Prompt] 共现表已加载: {len(_PMI_TABLE)} 个标签")
        except Exception as e:
            print(f"[Anima-Prompt] 共现表加载失败: {e}")
            _PMI_TABLE = {}
    else:
        print(f"[Anima-Prompt] 共现表不存在: {_PMI_PATH}")
        print("[Anima-Prompt] 请先运行 scripts/build_cooccurrence_pmi.py")
        _PMI_TABLE = {}
    return _PMI_TABLE


# ── System Prompt 文件管理 ──────────────────────────────────

_SP_DIR = Path(__file__).resolve().parent / "systemprompt"


def _list_sp_files():
    items = ["🛠️ 自定义"]
    if _SP_DIR.exists():
        for f in sorted(_SP_DIR.iterdir()):
            if f.is_file() and f.suffix in {".txt", ".md"}:
                items.append(f"📄 {f.name}")
    return items


def _read_sp_file(selection: str) -> str:
    if not selection or "🛠️" in selection or "自定义" in selection:
        return ""
    name = selection
    for p in ("📄", "📜"):
        if name.startswith(p):
            name = name[len(p):].strip()
            break
    fp = _SP_DIR / name
    return fp.read_text(encoding="utf-8").strip() if fp.exists() else ""


def _free_vram():
    gc.collect()
    try:
        model_management.soft_empty_cache()
    except:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass


# ── 角色特征精确查表 ──────────────────────────────────────
# 角色名是专有名词，语义搜索不可靠（keqing 会匹配到 qingyi）。
# 用 build_character_lookup.py 预生成的 JSON 做精确查表。

_CHAR_LOOKUP = None
_CHAR_LOOKUP_PATH = Path(__file__).resolve().parent / "data" / "character_features.json"


def _load_char_lookup(index_dir=None):
    """加载角色特征查找表（插件内置 data 目录）。index_dir 参数保留兼容，已不使用。"""
    global _CHAR_LOOKUP
    if _CHAR_LOOKUP is not None:
        return _CHAR_LOOKUP
    fp = _CHAR_LOOKUP_PATH
    if fp.exists():
        try:
            _CHAR_LOOKUP = json.loads(fp.read_text(encoding="utf-8"))
            print(f"[Anima-Prompt] 角色查找表已加载: {len(_CHAR_LOOKUP)} 个角色")
        except Exception as e:
            print(f"[Anima-Prompt] 角色查找表加载失败: {e}")
            _CHAR_LOOKUP = {}
    else:
        print(f"[Anima-Prompt] 角色查找表不存在: {fp}")
        _CHAR_LOOKUP = {}
    return _CHAR_LOOKUP


# 换装/形态信号：出现时跳过默认服装特征（见 system prompt §7.2）
_COSTUME_SIGNAL_WORDS = {
    "alternative costume", "official alternate costume", "cosplay", "crossover",
    "maid", "bunny", "swimsuit", "bikini", "kimono", "yukata", "china dress",
    "school uniform", "serafuku", "armor", "nurse", "wedding dress", "pajamas",
    "naked apron", "apron", "lingerie", "casual", "hoodie", "nude", "naked",
    "off shoulder", "dress", "uniform",
}

# 外貌类别关键词：用户写了某类，则查表里同类标签全部跳过（代码硬性覆盖）
# 发型类要覆盖不含 "hair" 字面的发型词（twintails/braid/bun/ponytail 等）
_APPEARANCE_CATEGORIES = {
    "hair": ["hair", "twintail", "twin tail", "braid", "bun", "ponytail", "bangs",
             "sidelocks", "ahoge", "hairband", "drill", "bob cut", "bowl cut"],
    "eyes": ["eyes", "eye ", "eyepatch", "heterochromia", "pupils"],
    "breasts": ["breasts"],
    "skin": ["skin", "dark-skinned", "pale"],
}

_EYE_COLOR_TAGS = {
    "blue eyes", "brown eyes", "green eyes", "red eyes", "purple eyes",
    "yellow eyes", "pink eyes", "aqua eyes", "cyan eyes", "teal eyes",
    "orange eyes", "grey eyes", "gray eyes", "black eyes", "white eyes",
    "violet eyes", "gold eyes", "golden eyes", "amber eyes", "silver eyes",
}


def _normalize_prompt_tag(tag: str) -> str:
    """统一为 prompt 使用的 tag 形态：小写、空格分隔。"""
    return tag.strip().lower().replace("_", " ")


def _is_eye_related_tag(tag: str) -> bool:
    """识别用户或筛选结果中的眼睛相关 tag。"""
    return _tag_category(_normalize_prompt_tag(tag)) == "eyes"


def _find_eye_color_tag(tags) -> str:
    """从角色外貌特征中找一个可补齐的瞳色 tag。"""
    for tag in tags:
        normalized = _normalize_prompt_tag(tag)
        if normalized in _EYE_COLOR_TAGS:
            return normalized
    return ""


def _append_unique_tag(out: list[str], seen: set[str], tag: str):
    normalized = _normalize_prompt_tag(tag)
    if normalized and normalized not in seen:
        seen.add(normalized)
        out.append(normalized)


def _extract_character_eye_tags(user_tags: list[str], lookup: dict) -> str:
    """
    单独输出角色名与眼睛 tag：
    - 有角色且有眼睛 tag：输出角色名 + 用户眼睛 tag；
    - 只有角色：从角色规范外貌补一个瞳色；
    - 只有眼睛 tag：只输出眼睛 tag，不补角色名；
    - 两者都没有：输出空字符串。
    """
    out = []
    seen = set()
    matched_entries = []
    eye_tags = []

    for raw in user_tags:
        key = raw.strip().lower().replace(" ", "_")
        entry = lookup.get(key) if lookup else None
        if entry is not None:
            display = entry.get("display", raw)
            matched_entries.append((display, entry))
        if _is_eye_related_tag(raw):
            eye_tags.append(raw)

    if not matched_entries and not eye_tags:
        return ""

    if matched_entries:
        for display, _entry in matched_entries:
            _append_unique_tag(out, seen, display)
        if eye_tags:
            for tag in eye_tags:
                _append_unique_tag(out, seen, tag)
        else:
            for _display, entry in matched_entries:
                eye_color = _find_eye_color_tag(entry.get("appearance", []))
                if eye_color:
                    _append_unique_tag(out, seen, eye_color)
    else:
        for tag in eye_tags:
            _append_unique_tag(out, seen, tag)

    return ", ".join(out)

def _conflicting_categories(user_tags):
    """检测用户输入涉及哪些外貌类别，这些类别的查表标签应跳过"""
    conflicts = set()
    for tag in user_tags:
        low = tag.lower()
        for cat, keywords in _APPEARANCE_CATEGORIES.items():
            if any(kw in low for kw in keywords):
                conflicts.add(cat)
    return conflicts


def _tag_category(tag):
    """判断一个外貌标签属于哪个类别（用于覆盖检测）"""
    low = tag.lower()
    for cat, keywords in _APPEARANCE_CATEGORIES.items():
        if any(kw in low for kw in keywords):
            return cat
    return None


def _extract_and_lookup_characters(query: str, lookup: dict):
    """
    从 query 中识别角色名并查表返回特征标签。
    返回 (feature_lines: list[str], matched_names: list[str])

    代码硬性覆盖：用户写了发型/发色/瞳色等 → 查表里同类标签自动剔除，
    保证用户能换发型换服装，查表不会强行覆盖用户意图。
    """
    if not lookup:
        return [], []

    # query 整体转小写，逗号分隔取候选片段
    segments = [s.strip() for s in query.replace("\n", ",").split(",") if s.strip()]

    # 检测用户是否提供了换装/服装信号（决定是否补默认服装）
    user_lower = query.lower()
    has_costume_signal = any(sig in user_lower for sig in _COSTUME_SIGNAL_WORDS)

    # 检测用户涉及的外貌类别（这些类别查表标签将被跳过）
    user_conflicts = _conflicting_categories(segments)

    feature_lines = []
    matched = []
    for seg in segments:
        key = seg.lower().replace(" ", "_")
        entry = lookup.get(key)
        if entry is None:
            continue
        name = entry.get("display", seg)
        matched.append(name)

        # 外貌：剔除与用户输入冲突类别的标签（代码硬性覆盖）
        app = entry.get("appearance", [])
        app_kept = [t for t in app if _tag_category(t) not in user_conflicts][:8]
        if app_kept:
            feature_lines.append(f"[角色规范:{name} 外貌] {', '.join(app_kept)}")

        # 仅当用户未提供换装信号时，才补充默认服装
        if not has_costume_signal:
            clo = entry.get("clothing", [])[:6]
            if clo:
                feature_lines.append(f"[角色规范:{name} 默认服装] {', '.join(clo)}")

    return feature_lines, matched



# 输出禁词：System Prompt 会再次约束，这里先从候选池移除，避免禁词进入参考上下文。
_FORBIDDEN_TAGS = {
    "masterpiece", "best_quality", "best quality", "highres", "absurdres",
    "score_9", "score_8_up", "score_7_up", "score_6_up", "score_5_up", "score_4_up",
    "newest", "very_newest", "year_2024", "year_2025", "year_2026",
    "source_anime", "source_pony", "8k", "4k", "hd", "ultra_high_resolution",
    "sunlight", "moonlight", "rim_light", "rim light", "backlighting",
    "warm_lighting", "warm lighting", "cool_lighting", "cool lighting",
    "god_rays", "god rays", "light_particles", "light particles",
    "volumetric_light", "volumetric light", "glowing", "illuminated", "spotlight", "flash",
}


def _is_forbidden_candidate(tag: str) -> bool:
    """PMI 候选进入 LLM 前的硬过滤；用户输入仍由 system prompt 和最终输出规则处理。"""
    low = tag.strip().lower()
    spaced = low.replace("_", " ")
    underscored = low.replace(" ", "_")
    return low in _FORBIDDEN_TAGS or spaced in _FORBIDDEN_TAGS or underscored in _FORBIDDEN_TAGS


_OUTPUT_FORBIDDEN_PATTERNS = [
    re.compile(r"\bsun(?:ny|lit|light|beam|rise|set)\b", re.IGNORECASE),
    re.compile(r"\b(?:moonlit|backlit|rim light|god rays?|light particles?|volumetric light)\b", re.IGNORECASE),
    re.compile(r"\b(?:vibrant|dramatic|cinematic)?\s*(?:atmosphere|ambience|ambiance|mood)\b", re.IGNORECASE),
    re.compile(r"\bclear sky\b", re.IGNORECASE),
]


def _is_forbidden_output_segment(segment: str) -> bool:
    """最终输出兜底过滤：PMI 已过滤候选，但 LLM 仍可能自行补违禁氛围/光照词。"""
    clean = segment.strip().strip(" .")
    if not clean:
        return True
    if _is_forbidden_candidate(clean):
        return True
    return any(p.search(clean) for p in _OUTPUT_FORBIDDEN_PATTERNS)


# ── 节点 1: 共现表标签扩充 ─────────────────────────────────

# 维度过滤词表：把共现扩充结果归类到 背景/物件/氛围。
# 共现表返回相关标签后，按这些关键词分流到对应维度。
_DIM_KEYWORDS = {
    "background": {  # 背景环境
        "indoor", "outdoor", "room", "wall", "floor", "window", "door", "building",
        "house", "architecture", "scenery", "tree", "bush", "plant", "grass", "fence",
        "sky", "cloud", "mountain", "forest", "street", "city", "interior", "background",
        "curtain", "bookshelf", "desk", "chair", "table", "bed", "stairs", "pillar",
    },
    "objects": {  # 场景物件
        "holding", "cup", "book", "flower", "bottle", "bag", "umbrella", "weapon",
        "sword", "tray", "phone", "instrument", "food", "fruit", "basket", "box",
        "lamp", "candle", "vase", "teacup", "plate", "bowl", "tool", "fan",
    },
    "atmosphere": {  # 氛围天气
        "rain", "fog", "mist", "steam", "snow", "petal", "dust", "particle", "light",
        "cloud", "wind", "leaf", "leaves", "blossom", "sparkle", "glow", "shadow",
        "night", "sunset", "dawn", "twilight", "fireflies", "smoke",
    },
}


# 泛标签停用词：作为扩充种子会引入同质化噪音（original 永远带出 shirt/skirt/panties）。
# 这些标签太泛，不用它们做 PMI 种子，但用户写的仍保留在最终输出。
_GENERIC_SEED_STOPWORDS = {
    "original", "1girl", "1boy", "solo", "2girls", "3girls", "multiple_girls",
    "1other", "male_focus", "female_focus",
}


def _expand_by_cooccurrence(tags, k=12, seed=0, exclude=None, drop_categories=None):
    """
    用共现 PMI 表扩充标签。从每个输入标签的 top30 候选池随机采样，
    既保证相关（都在 PMI top30 内）又有随机性（seed 控制，可复现）。

    drop_categories: 用户已指定的外貌类别（如 {hair, eyes}），扩充结果剔除同类，
                     避免和用户的 short hair / red eyes 冲突。

    返回去重后的扩充标签列表。
    """
    table = _load_pmi_table()
    if not table:
        return []
    exclude = set(exclude or [])
    exclude |= set(tags)
    drop_categories = drop_categories or set()

    # 累加每个输入标签的 PMI 候选（合并打分）
    score = {}
    for tag in tags:
        key = tag.strip().lower().replace(" ", "_")
        # 跳过泛标签种子（original/1girl 等），它们的共现是同质化噪音源
        if key in _GENERIC_SEED_STOPWORDS:
            continue
        candidates = table.get(key, [])
        for other, s in candidates:
            if other in exclude:
                continue
            if _is_forbidden_candidate(other):
                continue
            # 剔除与用户输入冲突类别（发型/瞳色等）的扩充标签
            if drop_categories and _tag_category(other.replace("_", " ")) in drop_categories:
                continue
            score[other] = max(score.get(other, 0.0), float(s))

    if not score:
        return []

    # 按分数排序成候选池，再从池中随机采样（相关 + 多变）
    ranked = sorted(score.items(), key=lambda x: -x[1])
    pool = [t for t, _ in ranked[:30]]
    rng = random.Random(seed if seed else 12345)
    n = min(k, len(pool))
    picked = rng.sample(pool, n) if len(pool) > n else pool
    # 下划线转空格，供 prompt 使用
    return [t.replace("_", " ") for t in picked]


def _filter_by_dimension(tags, dim):
    """把扩充标签按维度关键词过滤，归类到 背景/物件/氛围"""
    keywords = _DIM_KEYWORDS.get(dim, set())
    out = []
    for tag in tags:
        low = tag.lower()
        if any(kw in low for kw in keywords):
            out.append(tag)
    return out


class AnimaPMIExpand:
    """PMI 共现表扩充：角色查表 + 标签关联扩展"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {
                    "multiline": True, "default": "",
                    "placeholder": "输入场景描述或 tag ..."
                }),
                "expand_count": ("INT", {"default": 12, "min": 0, "max": 30, "step": 1,
                    "tooltip": "共现扩充标签数。从 PMI top30 候选池随机采样。"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1,
                    "tooltip": "随机种子。固定=可复现，变化=同输入出不同扩充组合（防同质化）。"}),
                "detail_background": ("BOOLEAN", {"default": True, "label": "细化背景"}),
                "add_objects": ("BOOLEAN", {"default": False, "label": "增加物件"}),
                "add_atmosphere": ("BOOLEAN", {"default": False, "label": "增加氛围"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("context_tags", "query", "character_eye_tags")
    FUNCTION = "search"
    CATEGORY = "PIC Pack/Anima"

    def search(self, query, expand_count, seed,
               detail_background, add_objects, add_atmosphere):
        if not query or not query.strip():
            return ("", query, "")

        # 把用户输入切成标签片段
        user_tags = [s.strip() for s in query.replace("\n", ",").split(",") if s.strip()]

        # ── 1. 角色名精确查表（专有名词不走语义搜索）──
        lookup = _load_char_lookup()
        character_eye_tags = _extract_character_eye_tags(user_tags, lookup)
        char_lines, matched_names = _extract_and_lookup_characters(query, lookup)
        if matched_names:
            print(f"[Anima-PMI] 命中角色: {', '.join(matched_names)}")

        # 用户涉及的外貌类别：共现扩充也要剔除同类（与角色查表覆盖逻辑一致）
        user_conflicts = _conflicting_categories(user_tags)

        # ── 2. 共现表扩充：从用户输入标签出发，PMI 关联扩充 ──
        expanded = _expand_by_cooccurrence(
            user_tags, k=expand_count, seed=seed, exclude=matched_names,
            drop_categories=user_conflicts,
        ) if expand_count > 0 else []

        # ── 3. 背景/物件/氛围维度：共现扩充结果按维度过滤 ──
        # 用更大的候选池做维度筛选（k 加倍，保证每维度有料）
        directed = {}
        if detail_background or add_objects or add_atmosphere:
            dim_pool = _expand_by_cooccurrence(
                user_tags, k=30, seed=seed + 1, exclude=matched_names,
                drop_categories=user_conflicts,
            )
            if detail_background:
                directed["背景环境"] = _filter_by_dimension(dim_pool, "background")[:6]
            if add_objects:
                directed["场景物件"] = _filter_by_dimension(dim_pool, "objects")[:6]
            if add_atmosphere:
                directed["氛围天气"] = _filter_by_dimension(dim_pool, "atmosphere")[:5]

        active = [k for k, v in {"细化背景": detail_background, "增加物件": add_objects,
                                 "增加氛围": add_atmosphere}.items() if v]
        if active:
            print(f"[Anima-PMI] 维度扩充: {', '.join(active)}")

        # ── 4. 合并：按三级处理强度分组（与 system prompt 的 LEVEL 框架对齐）──
        parts = []
        if char_lines:
            parts.append("=== 【LEVEL 1 必须使用】角色规范特征（用户覆盖除外）===")
            parts.extend(char_lines)
        for group_name, tags in directed.items():
            if tags:
                parts.append(f"=== 【LEVEL 2 建议采用】{group_name}（选 2-4 个）===")
                parts.extend(tags)
        if expanded:
            parts.append("=== 【LEVEL 3 自由发挥】关联标签参考（按需取舍）===")
            parts.extend(expanded)

        return ("\n".join(parts), query, character_eye_tags)


# ── 节点 2: Prompt 组装 ────────────────────────────────────

# NL 长度预设（借鉴 ToriiGate 的预设式控制）
# 每个选项注入不同的自然语言长度/详细度指令到 prompt 末尾
_NL_LENGTH_PRESETS = {
    "无 - 仅标签": (
        "Do NOT add any natural language sentence. Output ONLY comma-separated tags."
    ),
    "短 - 1句": (
        "MUST append exactly ONE natural-language sentence after all comma-separated tags. "
        "Describe spatial relation, pose, and one visible motion. No 'as if', no backstory, no inner feelings."
    ),
    "中 - 2-3句": (
        "MUST append 2-3 natural-language sentences after all comma-separated tags. "
        "Describe subject placement, pose/action, clothing texture, and scene details as visible image content. "
        "FORBIDDEN: 'as if' phrasing, invented backstory, inner feelings."
    ),
    "长 - 4-5句": (
        "MUST append 4-5 natural-language sentences after all comma-separated tags. "
        "Build a detailed visual paragraph covering composition, foreground/background relation, pose/action, expression, clothing materials, props, and environment. "
        "Use concrete visible details and present participles. FORBIDDEN: 'as if' phrasing, invented backstory, inner feelings, abstract moods."
    ),
    "超长 - 6-8句": (
        "MUST append 6-8 natural-language sentences after all comma-separated tags. "
        "Write a detailed visual description of the image: subject placement, camera framing, pose/action, expression, hair movement, clothing layers and textures, props, foreground/background relation, environment surfaces, and weather/atmospheric details when present. "
        "Each sentence should add new visible information. Keep it concrete and image-grounded. FORBIDDEN: 'as if' phrasing, invented backstory, inner feelings, abstract moods."
    ),
}


class AnimaPromptAssembler:
    """组装 System Prompt + 检索结果 + 用户输入"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt_source": (_list_sp_files(), {
                    "default": "🛠️ 自定义",
                    "label": "system prompt"
                }),
                                "rag_context": ("STRING", {
                    "forceInput": True, "default": "",
                    "label": "PMI context"
                }),
            },
            "optional": {
                "user_input": ("STRING", {
                    "forceInput": True, "default": "",
                    "label": "user input (auto from query)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("full_prompt",)
    FUNCTION = "assemble"
    CATEGORY = "PIC Pack/Anima"

    def assemble(self, system_prompt_source, rag_context, user_input):
        sp = _read_sp_file(system_prompt_source)
        if not sp:
            sp = "You are an Anima3 prompt engineer. Convert user input into 1 line English tags."

        parts = [sp]

        if rag_context.strip():
            parts.append(f"\n\n[PMI扩充参考]:\n{rag_context}")

        parts.append(f"\n\n[用户输入]:\n{user_input}")

        parts.append(
            "\n\n---\n"
            "TAG HANDLING — three strictness levels:\n"
            "\n"
            "【LEVEL 1 · MANDATORY · 不可改】\n"
            "- [用户输入] is absolute truth semantically. Use all user-provided tags, but normalize final tag spelling to Anima official format: lowercase, spaces instead of underscores.\n"
            "- '角色规范特征' (canonical character tags): use them EXACTLY (e.g. purple hair, twintails).\n"
            "  Do NOT invent other hair/eye colors. EXCEPTION: if user gave a conflicting trait, user wins.\n"
            "\n"
            "【LEVEL 2 · RECOMMENDED · 建议采用】\n"
            "- '背景环境/场景物件/氛围天气' groups: pick 2-4 that FIT the scene. These shape the setting.\n"
            "\n"
            "【LEVEL 3 · FREE · 自由发挥】\n"
            "- '关联标签参考': optional pool. Freely pick what fits, freely skip what doesn't.\n"
            "- You MAY add your own fitting tags for pose/expression/camera not in any list.\n"
            "\n"
            "【NATURAL LANGUAGE · 结尾】\n"
            "- END with natural language weaving the tags into a visual scene (length set by later instructions).\n"
            "- FORBIDDEN: 'as if...' phrasing, invented backstory, inner feelings.\n"
            "\n"
            "Output exactly 1 line. Use lowercase tags with spaces, not underscores (score tags are the only underscore exception). No markdown. No explanation. No thinking."
        )

        return ("\n".join(parts),)


# ── 节点 3: 纯文本 GGUF LLM ────────────────────────────────

# 冲突检测兜底（代码化 system prompt §3.1，LLM 选完后做硬约束清理）
# 互斥对：成对标签不能共存，保留先出现的（通常是用户/角色特征，优先级高）。
_CONFLICT_PAIRS = [
    # 视角互斥
    ("from front", "from behind"), ("from above", "from below"),
    ("looking at viewer", "facing away"),
    # 动作互斥（同一时刻身体状态）
    ("standing", "lying"), ("standing", "sitting"), ("sitting", "lying"),
    ("standing", "on back"), ("standing", "kneeling"),
]
# 互斥组：同组内只能留一个（同部位的互斥状态，如盔甲覆盖度）
_CONFLICT_GROUPS = [
    {"completely nude", "full armor", "bikini armor", "school uniform", "maid"},  # 上身覆盖度
    {"long hair", "short hair", "very long hair"},  # 发长（角色覆盖后通常已唯一）
]


def _resolve_conflicts(tag_part):
    """
    对 LLM 输出的标签部分做冲突清理。保留先出现的标签（优先级高），
    删除与之冲突的后续标签。返回清理后的标签字符串。
    """
    tags = [t.strip() for t in tag_part.split(",") if t.strip()]
    tags_low = [t.lower() for t in tags]
    removed = set()

    # 互斥对：若两个都在，删后出现的
    for a, b in _CONFLICT_PAIRS:
        if a in tags_low and b in tags_low:
            ia, ib = tags_low.index(a), tags_low.index(b)
            removed.add(max(ia, ib))

    # 互斥组：组内出现多个，只留第一个
    for group in _CONFLICT_GROUPS:
        present = [i for i, t in enumerate(tags_low) if t in group]
        for idx in present[1:]:
            removed.add(idx)

    kept = [t for i, t in enumerate(tags) if i not in removed]
    if removed:
        dropped = [tags[i] for i in sorted(removed)]
        print(f"[PIC-LLM] 冲突清理，删除: {', '.join(dropped)}")
    return ", ".join(kept)


def _extract_weighted_tags(text):
    """从文本提取 A1111 权重语法标签：(tag:1.5) 或 ((tag)) 或 [tag]"""
    if not text:
        return []
    found = []
    # (tag:1.5) 形式
    found += re.findall(r'\([^()]*:\s*[\d.]+\)', text)
    # ((tag)) 或 (tag) 加强（含嵌套括号但非角色名）
    found += re.findall(r'\({2,}[^()]+\){2,}', text)
    return found


def _protect_weighted_syntax(output, user_input):
    """
    保护用户输入的权重语法：若 LLM 输出丢失了用户写的 (tag:1.5)，
    把丢失的权重标签补回输出开头。代码兜底，不靠 LLM 遵守。
    """
    user_weights = _extract_weighted_tags(user_input)
    if not user_weights:
        return output
    missing = []
    for w in user_weights:
        # 提取权重标签的核心词（去括号和数字），检查输出是否已含
        core = re.sub(r'[():\d.]+', '', w).strip().lower()
        if w not in output and core and core not in output.lower():
            missing.append(w)
        elif w not in output and core in output.lower():
            # 核心词在但权重语法丢了 → 把裸词替换回带权重的形式
            output = re.sub(r'\b' + re.escape(core) + r'\b', w, output, count=1, flags=re.IGNORECASE)
    if missing:
        # 补回完全丢失的权重标签
        output = ", ".join(missing) + ", " + output
        print(f"[PIC-LLM] 权重语法保护，补回: {', '.join(missing)}")
    return output


def _extract_user_input_from_prompt(system_prompt: str) -> str:
    """从组装后的 full prompt 中取真实用户输入，避免把规则/示例里的权重语法当成用户标签。"""
    if not system_prompt:
        return ""
    m = re.search(r"\[(?:用户输入|user input)[^\]]*\]\s*[:：]\s*(.*?)(?:\n\s*\n---|\Z)",
                  system_prompt, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_literal_user_tags(user_input: str) -> list[str]:
    if not user_input:
        return []
    tags = []
    for raw in re.split(r"[,\n]", user_input):
        tag = raw.strip()
        if not tag or len(tag) > 80:
            continue
        if re.search(r"[.!?;；。！？]", tag):
            continue
        tags.append(tag)
    return tags


def _strip_unrequested_weight_syntax(output: str, user_input: str) -> str:
    """只允许用户亲自输入的 A1111 权重语法；LLM 自己发明的权重/占位符删掉。"""
    allowed = set(_extract_weighted_tags(user_input))

    def keep_or_drop(m):
        token = m.group(0)
        return token if token in allowed else ""

    output = re.sub(r"\([^()]*:\s*[\d.]+\)", keep_or_drop, output)
    output = re.sub(r"\({2,}[^()]+\){2,}", keep_or_drop, output)
    output = re.sub(r"\s*,\s*,+", ",", output)
    return output.strip(" ,")


def _normalize_phrase_segment(segment: str) -> list[str]:
    """把 LLM 偶发的短英文短语压回 tag 形态，例如 wearing dress / keqing with purple hair。"""
    seg = segment.strip()
    if not seg:
        return []
    m = re.match(r"^wearing\s+(.+)$", seg, flags=re.IGNORECASE)
    if m:
        return [m.group(1).strip()]
    m = re.match(r"^[a-z0-9_()' -]{2,40}\s+with\s+(.+)$", seg, flags=re.IGNORECASE)
    if m:
        return [p.strip() for p in re.split(r"\s+and\s+|/", m.group(1)) if p.strip()]
    return [seg]


def _sanitize_generated_output(output: str) -> str:
    kept = []
    seen = set()
    for raw in output.split(","):
        for seg in _normalize_phrase_segment(raw):
            seg = seg.strip().strip(" .")
            if not seg:
                continue
            if re.fullmatch(r"\(+\s*tag\s*\)+", seg, flags=re.IGNORECASE):
                continue
            if _is_forbidden_output_segment(seg):
                continue
            key = seg.lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(seg)
    return ", ".join(kept)


def _restore_user_literal_tags(output: str, user_input: str) -> str:
    """用户输入是硬约束；LLM 漏掉的短 tag 补回开头，最终再统一为 Anima 官方格式。"""
    missing = []
    low = output.lower()
    for tag in _extract_literal_user_tags(user_input):
        if _extract_weighted_tags(tag):
            continue
        if tag.lower() not in low and _format_anima_tag_segment(tag).lower() not in low:
            missing.append(tag)
    if missing:
        output = ", ".join(missing) + (", " + output if output else "")
    return output


def _format_anima_tag_segment(segment: str) -> str:
    """按 Anima 官方模型卡格式化单个 tag：普通 tag 用空格，不用下划线；score 标签例外。"""
    tag = segment.strip().lower()
    if not tag:
        return ""
    if re.fullmatch(r"score_\d+(?:_up)?", tag):
        return tag
    tag = tag.replace("_", " ")
    tag = re.sub(r"\s+", " ", tag)
    tag = tag.replace(" ( ", " (").replace(" )", ")")
    tag = re.sub(r"\(\s+", "(", tag)
    tag = re.sub(r"\s+\)", ")", tag)
    tag = re.sub(r"\s+([,.;:!?])", r"\1", tag)
    tag = re.sub(r"([(:])\s+", r"\1", tag)
    return tag.strip()


def _format_anima_official_tags(output: str) -> str:
    """最终输出兜底：统一 lowercase、逗号空格，并把 Danbooru 下划线 tag 转为 Anima 官方空格写法。"""
    kept = []
    seen = set()
    for raw in output.split(","):
        tag = _format_anima_tag_segment(raw)
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(tag)
    return ", ".join(kept)


def _nl_is_requested(system_prompt: str) -> bool:
    """StyleDirector 非“仅标签”档时，最终输出必须带自然语言句子。"""
    if not system_prompt or "[NATURAL LANGUAGE LENGTH]" not in system_prompt:
        return False
    return "Do NOT add any natural language sentence" not in system_prompt


def _strengthen_user_message_for_nl(user_message: str, system_prompt: str) -> str:
    """避免旧工作流 user_message 里的 only tags 覆盖自然语言长度指令。"""
    msg = (user_message or "Generate.").strip()
    if _nl_is_requested(system_prompt):
        msg = re.sub(r"\boutput only one line of tags\b", "output one line prompt", msg, flags=re.IGNORECASE)
        msg += " Follow NATURAL LANGUAGE LENGTH exactly; append the required final natural-language paragraph after the tags."
    return msg


def _looks_like_natural_language(segment: str) -> bool:
    """粗判末尾是否已经是自然语言，而不是普通 tag。"""
    words = re.findall(r"[a-z]+", segment.lower())
    if len(words) < 7:
        return False
    verbs = {"is", "are", "stands", "sits", "lies", "leans", "holds", "faces", "moves", "wears", "drifts", "frames", "rests", "fills"}
    links = {"while", "with", "as", "near", "inside", "around", "beside", "against", "through", "across"}
    return bool((verbs | links) & set(words))


def _requested_nl_sentence_count(system_prompt: str) -> int:
    """从 StyleDirector 注入文本推断最低自然语言句数。"""
    if not _nl_is_requested(system_prompt):
        return 0
    if "6-8 natural-language sentences" in system_prompt:
        return 6
    if "4-5 natural-language sentences" in system_prompt:
        return 4
    if "2-3 natural-language sentences" in system_prompt:
        return 2
    return 1


def _count_natural_language_sentences(output: str) -> int:
    """粗略统计已存在的自然语言句子数，用于长描述补足。"""
    parts = [p.strip() for p in output.split(",") if p.strip()]
    nl_parts = [p for p in parts if _looks_like_natural_language(p)]
    if not nl_parts:
        return 0
    text = " ".join(nl_parts)
    sentences = [s for s in re.split(r"[.!?]+\s*", text) if len(re.findall(r"[a-z]+", s.lower())) >= 6]
    return max(1, len(sentences))


def _append_fallback_natural_language(output: str, system_prompt: str) -> str:
    """LLM 自然语言太短时，用已有 tags 兜底补足到当前长度档。"""
    target = _requested_nl_sentence_count(system_prompt)
    if target <= 0:
        return output
    existing = _count_natural_language_sentences(output)
    if existing >= target:
        return output

    parts = [p.strip() for p in output.split(",") if p.strip()]
    if not parts:
        return output

    low_parts = [p.lower() for p in parts]
    plural = any(t in low_parts for t in {"2girls", "2boys", "multiple girls", "multiple boys"})
    subject = "the characters" if plural else "the character"

    scene_tags = {
        "bedroom", "classroom", "forest", "beach", "street", "garden", "rooftop", "stage",
        "library", "office", "kitchen", "shrine", "cafe", "school", "city", "indoors", "outdoors"
    }
    pose_tags = {
        "standing", "sitting", "lying", "kneeling", "walking", "running", "dancing",
        "singing", "holding microphone", "holding weapon", "looking at viewer", "facing viewer"
    }
    scene = next((t for t in low_parts if t in scene_tags), "the scene")
    scene_phrases = {
        "stage": "on stage", "street": "on the street", "rooftop": "on the rooftop",
        "beach": "on the beach", "garden": "in the garden", "forest": "in the forest",
        "classroom": "in the classroom", "bedroom": "in the bedroom", "library": "in the library",
        "office": "in the office", "kitchen": "in the kitchen", "shrine": "at the shrine",
        "cafe": "in the cafe", "city": "in the city", "indoors": "indoors", "outdoors": "outdoors",
    }
    scene_phrase = scene_phrases.get(scene, scene if scene == "the scene" else f"in {scene}")
    scene_detail_phrases = {"stage": "the stage area", "street": "the street", "rooftop": "the rooftop", "beach": "the beach", "garden": "the garden", "forest": "the forest", "classroom": "the classroom", "bedroom": "the bedroom", "library": "the library", "office": "the office", "kitchen": "the kitchen", "shrine": "the shrine grounds", "cafe": "the cafe", "city": "the city", "indoors": "the interior", "outdoors": "the outdoor setting"}
    scene_detail = scene_detail_phrases.get(scene, "the setting")
    pose = next((t for t in low_parts if t in pose_tags), "holds the pose")
    if pose in {"standing", "sitting", "lying", "kneeling", "walking", "running", "dancing", "singing"}:
        pose_text = f"is {pose}"
    else:
        pose_text = pose

    appearance = next((t for t in low_parts if any(k in t for k in ["hair", "eyes", "horns", "ears"])), "the visible character details")
    clothing = next((t for t in low_parts if any(k in t for k in ["dress", "shirt", "skirt", "uniform", "kimono", "armor", "jacket", "bodysuit"])), "the clothing")
    prop = next((t for t in low_parts if any(k in t for k in ["holding", "microphone", "weapon", "book", "umbrella", "sword", "flower"])), "nearby props")
    camera = next((t for t in low_parts if any(k in t for k in ["shot", "view", "from front", "full body", "cowboy shot"])), "the camera framing")

    fallback_sentences = [
        f"{subject} {pose_text} {scene_phrase} while {appearance} anchors the first impression.",
        f"{clothing} defines the silhouette with clear folds and layered edges around the body.",
        f"{camera} keeps the pose readable and leaves enough space for the surrounding environment.",
        f"{prop} provide concrete focal details instead of leaving the background empty.",
        "foreground and background elements frame the figure so the viewer can read the space at a glance.",
        f"small surface details within {scene_detail} make the image feel more specific and fully staged.",
        "the final composition balances character detail with the setting so neither part feels isolated.",
        "the image holds on visible motion and texture rather than implied story or inner emotion.",
    ]
    needed = max(0, target - existing)
    paragraph = " ".join(fallback_sentences[:needed])
    return output + ", " + paragraph if paragraph else output

from llama_cpp import Llama
import folder_paths

# 全局 model cache
_LLM = None
_LLM_PATH = None


def _list_gguf_models():
    """列出 models/LLM/ 中的 GGUF 文件（排除 mmproj）"""
    try:
        files = folder_paths.get_filename_list("LLM")
        return [f for f in files if "mmproj" not in f.lower()]
    except:
        return [""]


class PIC_GGUFTextLLM:
    """加载 GGUF 模型，接收 system prompt + user message，输出文本"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_list_gguf_models(), {"label": "GGUF model"}),
                "user_message": ("STRING", {
                    "multiline": False, "default": "Generate the Anima prompt now. Follow NATURAL LANGUAGE LENGTH exactly.",
                    "label": "user message"
                }),
                "max_tokens": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "repeat_penalty": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 1.5, "step": 0.01,
                    "tooltip": "重复惩罚。>1 抑制重复词。1.1-1.2 适合防止 green green green 循环。"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1,
                    "control_after_generate": True}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 256, "step": 1,
                    "tooltip": "-1 = all layers on GPU. 0 = CPU only. Try 20-30 if GPU errors occur."}),
                "n_ctx": ("INT", {"default": 8192, "min": 2048, "max": 32768, "step": 512}),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "forceInput": True, "default": "",
                    "label": "system prompt"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "generate"
    CATEGORY = "PIC Pack/Anima"

    def generate(self, model, user_message,
                 max_tokens, temperature, top_p, repeat_penalty, seed,
                 n_gpu_layers, n_ctx, unload_after, system_prompt=""):
        global _LLM, _LLM_PATH

        if not model or not model.strip():
            return ("Error: no model selected",)

        # 加载模型
        model_path = str(Path(folder_paths.get_folder_paths("LLM")[0]) / model)
        if not Path(model_path).exists():
            return (f"Error: model not found: {model_path}",)

        if _LLM is None or _LLM_PATH != model_path:
            self._unload()

            print(f"[PIC-LLM] Loading: {model} (gpu_layers={n_gpu_layers}, ctx={n_ctx})")
            _LLM = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                seed=seed,
                verbose=False,
            )
            _LLM_PATH = model_path

        # 生成
        messages = []
        if system_prompt and system_prompt.strip():
            # 在 system prompt 最前面注入 "禁止思考" 指令
            sp = "IMPORTANT: Output ONLY the final result. Do NOT output thinking, reasoning, or explanation. Direct output only.\n\n" + system_prompt
            messages.append({"role": "system", "content": sp})
        messages.append({"role": "user", "content": _strengthen_user_message_for_nl(user_message, system_prompt)})

        print(f"[PIC-LLM] Generating... (temp={temperature}, max_t={max_tokens}, rep={repeat_penalty})")
        resp = _LLM.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )

        output = resp["choices"][0]["message"]["content"].strip()

        # ── 剥离思考和元文本 ──────────────────────────────
        import re
        # 1. 彻底切除 <think>...</think> 整块（含内容），不只是标签
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL | re.IGNORECASE)
        output = re.sub(r'<reasoning>.*?</reasoning>', '', output, flags=re.DOTALL | re.IGNORECASE)
        # 2. 若只有开标签无闭合（被 max_tokens 截断），切掉开标签之后所有内容前的部分
        #    取最后一个孤立 </think> 之后的内容（思考结束后才是答案）
        if '</think>' in output.lower():
            output = re.split(r'</think>', output, flags=re.IGNORECASE)[-1]
        # 3. 残留的孤立标签清理
        output = re.sub(r'<\s*/?\s*think\s*>', '', output, flags=re.IGNORECASE)
        output = re.sub(r'<\s*/?\s*reasoning\s*>', '', output, flags=re.IGNORECASE)
        output = output.strip()

        # 按行处理，过滤掉元文本行
        lines = output.split('\n')
        clean_lines = []
        meta_patterns = [
            r'^let\s*(me|us|["\']?s)\b',     # Let me / Let's / let us
            r'^i[ "\']?(ll|will|would)\b',    # I'll / I will / I would
            r'^here[ "\']?s\b',                # Here's
            r'^(this|that|it)\s+is\b',         # This is / That is
            r'^(the|a|an)\s+\w+\s+(prompt|output|result|NL)\b',  # The prompt/output/result/NL
            r'^notice\b',                      # Notice that...
            r'^note\b',                        # Note:
            r'^(according|based|following)\b', # According to / Based on
            r'^output\b',                      # Output: (standalone, not as content)
            r'^prompt\s*:',                    # Prompt:
            r'^(the|a)\s+(correct|final|actual|generated)\b',  # The correct/final prompt
        ]
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            is_meta = False
            for pat in meta_patterns:
                if re.match(pat, stripped, re.IGNORECASE):
                    is_meta = True
                    break
            if not is_meta:
                clean_lines.append(stripped)

        if clean_lines:
            # 如果还有多行，取最长的那行（最可能是完整 prompt）
            if len(clean_lines) > 1:
                clean_lines.sort(key=len, reverse=True)
            output = clean_lines[0]
        else:
            output = ""

        # ── 剥离 markdown 格式化符号 ──────────────────────
        # 去掉 code fence / backtick 包裹
        output = re.sub(r'^```\w*\s*', '', output)
        output = re.sub(r'\s*```$', '', output)
        output = output.strip('`').strip()
        # 去掉行首的 markdown 列表符号 (* - 1.)
        output = re.sub(r'^[\*\-]\s+', '', output)
        output = re.sub(r'^\d+\.\s+', '', output)
        # 去掉行首 > 引用符号
        output = re.sub(r'^>\s+', '', output)
        # 去掉 **bold** 和 *italic* 标记（保留内部文字）
        output = re.sub(r'\*\*([^*]+)\*\*', r'\1', output)
        output = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', output)
        # 去掉行内的 `code` 标记
        output = re.sub(r'`([^`]+)`', r'\1', output)
        # ── 清理标记词泄漏：LLM 把指令里的标签名当内容输出 ──
        # 如 "natural language:" "[NATURAL LANGUAGE]:" 等元标记
        output = re.sub(r'\b(?:natural language|nl)\s*[:：]\s*', '', output, flags=re.IGNORECASE)
        output = re.sub(r'\[[^\]]*\]\s*[:：]?\s*', '', output)  # 残留的 [XXX] 标记
        # 清理 "over her default outfit" 这类从指令泄漏的元描述
        output = re.sub(r'\s*(?:over|with|in)\s+(?:her|his|their)\s+default\s+(?:outfit|clothing|costume)', '', output, flags=re.IGNORECASE)
        # 清理多余的空白
        output = re.sub(r' +', ' ', output)
        output = output.strip()

        # ── 清理 LLM 犹豫标签：括号注释如 "(no gloves)" "(implicit)" "(implied...)" ──
        # 移除标签后的括号补充说明，但保留 IP 角色名的括号如 keqing_(genshin_impact)
        # 和权重语法 (close-up:1.5)
        # 策略1：删除含 犹豫词/元描述词 的括号（default/override/pose/implicit 等）
        output = re.sub(
            r'\s*\((?=[^)]*\b(?:no|implicit|implied|general|style|the pose|default|override|implied by)\b)[^)]*\)',
            '', output, flags=re.IGNORECASE)
        # 策略2：删除"裸词复述"括号 —— 括号内不含冒号(非权重)、不含 IP 关键词、
        # 且是单个短词(LLM 复述维度标记如 "hospital gown (hospital)")
        def _strip_echo_paren(m):
            inner = m.group(1)
            # 含冒号=权重语法，含下划线/常见IP后缀=角色名，保留
            if ":" in inner or "_" in inner or any(
                kw in inner.lower() for kw in ["impact", "fate", "vocaloid", "zero", "frontline", "rail"]):
                return m.group(0)
            return ""  # 其余括号（裸词复述）删除
        output = re.sub(r'\s*\(([^)]{1,25})\)', _strip_echo_paren, output)

        # ── 最终输出兜底：只以真实用户输入为硬约束，清掉 LLM 自行发明的权重/光照/抽象氛围词 ──
        user_input_for_cleanup = _extract_user_input_from_prompt(system_prompt)
        output = _strip_unrequested_weight_syntax(output, user_input_for_cleanup)
        output = _sanitize_generated_output(output)
        output = _restore_user_literal_tags(output, user_input_for_cleanup)
        # ── 官方格式化 + 冲突清理兜底（代码化 §3.1，LLM 选完后做硬约束）──
        output = _format_anima_official_tags(output)
        output = _resolve_conflicts(output)
        output = _format_anima_official_tags(output)

        # ── 权重语法保护：用户写的 (tag:1.5) 若被 LLM 删了，补回 ──
        output = _protect_weighted_syntax(output, user_input_for_cleanup)
        output = _append_fallback_natural_language(output, system_prompt)

        print(f"[PIC-LLM] Done: {len(output)} chars")

        if unload_after:
            self._unload()

        _free_vram()
        return (output,)

    @staticmethod
    def _unload():
        global _LLM, _LLM_PATH
        if _LLM is not None:
            del _LLM
            _LLM = None
            _LLM_PATH = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[PIC-LLM] Unloaded")


# ── 注册 ──────────────────────────────────────────────────

# ── 节点 4: 风格控制 ──────────────────────────────────────

# 背景/物件/氛围已移到检索节点（定向检索真实标签，更可靠）。
# 这里只保留生成阶段才能控制的维度：动态表现、随机风格。
_STYLE_PRESETS = {
    "enhance_motion": (
        "EMPHASIZE DYNAMIC MOTION. Use pose tags with active movement (swaying, striding, "
        "leaning, reaching). Add motion-related tags: motion lines, wind-blown hair, "
        "floating hair, fluttering fabric, speed lines. "
        "Natural language must describe the movement and its effect on clothing and hair."
    ),
    "vary_style": (
        "VARY the overall ART STYLE for this generation. Pick a different mood "
        "and aesthetic direction from the retrieved tags. Try a different camera angle "
        "or composition than usual. (Note: requires seed=randomize or higher temperature to take real effect.)"
    ),
}

class PIC_StyleDirector:
    """在 prompt 末尾追加风格控制指令"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "full_prompt": ("STRING", {
                    "forceInput": True, "default": "",
                    "label": "full prompt"
                }),
                "nl_length": (list(_NL_LENGTH_PRESETS.keys()), {
                    "default": "超长 - 6-8句",
                    "label": "自然语言长度"
                }),
                "enhance_motion": ("BOOLEAN", {"default": False, "label": "加强动态"}),
                "vary_style": ("BOOLEAN", {"default": False, "label": "随机风格"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("full_prompt",)
    FUNCTION = "direct"
    CATEGORY = "PIC Pack/Anima"

    def direct(self, full_prompt, nl_length, enhance_motion, vary_style):
        toggles = {
            "enhance_motion": enhance_motion,
            "vary_style": vary_style,
        }

        extras = []

        # NL 长度指令始终注入（主控，借鉴 ToriiGate 预设式控制）
        legacy_map = {
            "中 - 1-2句": "中 - 2-3句",
            "长 - 2-3句": "长 - 4-5句",
        }
        resolved_nl_length = legacy_map.get(nl_length, nl_length)
        nl_rule = _NL_LENGTH_PRESETS.get(resolved_nl_length, _NL_LENGTH_PRESETS["超长 - 6-8句"])
        extras.append(f"\n\n[NATURAL LANGUAGE LENGTH]:\n{nl_rule}")

        # 风格开关指令按需追加
        active = [k for k, v in toggles.items() if v]
        if active:
            extras.append("\n\n[ADDITIONAL STYLE INSTRUCTIONS — these OVERRIDE defaults]:")
            for key in active:
                extras.append(f"- {_STYLE_PRESETS[key]}")

        print(f"[PIC-StyleDirector] NL={resolved_nl_length}, Active: {', '.join(active) if active else '无'}")
        return (full_prompt + "\n".join(extras),)


# ── 注册 ──────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "PIC_AnimaPMIExpand": AnimaPMIExpand,
    "PIC_AnimaRAGSearch": AnimaPMIExpand,  # legacy workflow compatibility
    "PIC_AnimaPromptAssembler": AnimaPromptAssembler,
    "PIC_GGUFTextLLM": PIC_GGUFTextLLM,
    "PIC_StyleDirector": PIC_StyleDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIC_AnimaPMIExpand": "PIC - Anima PMI Expand",
    "PIC_AnimaRAGSearch": "PIC - Anima PMI Expand (legacy id)",
    "PIC_AnimaPromptAssembler": "PIC - Anima Prompt Assembler",
    "PIC_GGUFTextLLM": "PIC - GGUF Text LLM",
    "PIC_StyleDirector": "PIC - Style Director",
}



