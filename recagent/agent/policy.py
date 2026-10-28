"""演示模式的有限题材词典；无法解析的约束显式保留。"""

from __future__ import annotations

import re
from dataclasses import dataclass

GENRE_ALIASES = {
    "科幻": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "喜剧": "Comedy",
    "comedy": "Comedy",
    "恐怖": "Horror",
    "horror": "Horror",
    "动作": "Action",
    "action": "Action",
    "剧情": "Drama",
    "drama": "Drama",
    "动画": "Animation",
    "animation": "Animation",
    "冒险": "Adventure",
    "adventure": "Adventure",
    "悬疑": "Mystery",
    "mystery": "Mystery",
    "爱情": "Romance",
    "romance": "Romance",
}
NEGATIVE = ("不要", "不看", "不想看", "别看", "避开", "排除")


@dataclass(frozen=True, slots=True)
class Preferences:
    genres: tuple[str, ...]
    exclude_genres: tuple[str, ...]
    unparsed_constraints: tuple[str, ...]


def parse_preferences(query: str) -> Preferences:
    """仅理解明确的题材词；未知限制不会被假装已执行。"""
    normalized = query.casefold()
    includes: set[str] = set()
    excludes: set[str] = set()
    for alias, genre in GENRE_ALIASES.items():
        for match in re.finditer(re.escape(alias), normalized):
            prefix = normalized[max(0, match.start() - 5) : match.start()]
            if any(prefix.endswith(word) for word in NEGATIVE):
                excludes.add(genre)
            else:
                includes.add(genre)
    includes.difference_update(excludes)
    unparsed: list[str] = []
    for match in re.finditer(r"(?:不要|不看|不想看|别看|避开|排除)([^，。,.；;]+)", normalized):
        phrase = match.group(1).strip()
        if phrase and not any(alias in phrase for alias in GENRE_ALIASES):
            unparsed.append(phrase)
    return Preferences(tuple(sorted(includes)), tuple(sorted(excludes)), tuple(unparsed))
