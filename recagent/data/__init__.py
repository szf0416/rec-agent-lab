"""数据层：下载、校验、加载、切分、负采样。

设计约定（详见 docs/ARCHITECTURE.md 第 2.1 节）：
- 原始数据只读，处理产物统一写到 processed 目录
- 切分与负采样必须保证无未来信息泄漏，泄露检查是硬性约束
"""

from recagent.data.download import (
    ML1M_SPEC,
    SPECS,
    DatasetSpec,
    DownloadReport,
    FileSpec,
    ensure_dataset,
    get_spec,
    md5_of,
    verify_dataset,
    verify_file,
)
from recagent.data.exceptions import (
    DataError,
    DownloadError,
    IntegrityError,
    LeakageError,
    ParseError,
    RecAgentError,
    SplitError,
)
from recagent.data.loading import (
    DatasetSplit,
    Interaction,
    build_sequences,
    from_ratings,
    from_ratings_positive,
    group_by_user,
    item_universe,
    sequence_stats,
    sort_by_time,
    to_dataframe,
    user_universe,
)
from recagent.data.ml1m import (
    Movie,
    MovieLens1M,
    Rating,
    User,
    export_metadata,
    load_ml1m,
    load_ml1m_cached,
)
from recagent.data.negative import (
    NegativeSample,
    NegativeSampler,
    build_candidate_pool,
    item_frequencies,
)
from recagent.data.splits import (
    LeaveOneOutSplitter,
    Splitter,
    TemporalSplitter,
    available_items_before,
    build_splitter,
    leakage_report,
    validate_no_leakage,
)

__all__ = [
    "ML1M_SPEC",
    "SPECS",
    "DatasetSpec",
    "DatasetSplit",
    "DataError",
    "DownloadError",
    "DownloadReport",
    "FileSpec",
    "IntegrityError",
    "Interaction",
    "LeaveOneOutSplitter",
    "LeakageError",
    "Movie",
    "MovieLens1M",
    "NegativeSample",
    "NegativeSampler",
    "ParseError",
    "Rating",
    "RecAgentError",
    "SplitError",
    "Splitter",
    "TemporalSplitter",
    "User",
    "available_items_before",
    "build_candidate_pool",
    "build_sequences",
    "build_splitter",
    "ensure_dataset",
    "export_metadata",
    "from_ratings",
    "from_ratings_positive",
    "get_spec",
    "group_by_user",
    "item_frequencies",
    "item_universe",
    "leakage_report",
    "load_ml1m",
    "load_ml1m_cached",
    "md5_of",
    "sequence_stats",
    "sort_by_time",
    "to_dataframe",
    "user_universe",
    "validate_no_leakage",
    "verify_dataset",
    "verify_file",
]
