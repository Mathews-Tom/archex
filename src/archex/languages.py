"""Language support registry and tier metadata."""

from __future__ import annotations

from dataclasses import dataclass

from archex.models import LanguageTier


@dataclass(frozen=True, slots=True)
class LanguageSupport:
    """Declared parser capability for one language identifier."""

    language_id: str
    display_name: str
    extensions: tuple[str, ...]
    tier: LanguageTier
    pack_name: str
    chunk_node_types: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.tier is LanguageTier.STRUCTURED and not self.chunk_node_types:
            raise ValueError("STRUCTURED languages must declare outline chunk_node_types")


_FULL = LanguageTier.FULL
_STRUCTURED = LanguageTier.STRUCTURED
_CHUNK = LanguageTier.CHUNK_ONLY
UNKNOWN_LANGUAGE_ID = "unknown"
UNKNOWN_LINE_WINDOW = 80

LANGUAGE_SUPPORT: dict[str, LanguageSupport] = {
    "python": LanguageSupport("python", "Python", (".py",), _FULL, "python"),
    "javascript": LanguageSupport("javascript", "JavaScript", (".js", ".jsx"), _FULL, "javascript"),
    "typescript": LanguageSupport("typescript", "TypeScript", (".ts", ".tsx"), _FULL, "typescript"),
    "tsx": LanguageSupport("tsx", "TSX", (".tsx",), _FULL, "tsx"),
    "go": LanguageSupport("go", "Go", (".go",), _FULL, "go"),
    "rust": LanguageSupport("rust", "Rust", (".rs",), _FULL, "rust"),
    "java": LanguageSupport("java", "Java", (".java",), _FULL, "java"),
    "kotlin": LanguageSupport("kotlin", "Kotlin", (".kt", ".kts"), _FULL, "kotlin"),
    "csharp": LanguageSupport("csharp", "C#", (".cs",), _FULL, "csharp"),
    "swift": LanguageSupport("swift", "Swift", (".swift",), _FULL, "swift"),
    "c": LanguageSupport("c", "C", (".c", ".h"), _FULL, "c"),
    "cpp": LanguageSupport(
        "cpp", "C++", (".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"), _FULL, "cpp"
    ),
    "php": LanguageSupport("php", "PHP", (".php",), _FULL, "php"),
    "ruby": LanguageSupport("ruby", "Ruby", (".rb",), _FULL, "ruby"),
    "scala": LanguageSupport("scala", "Scala", (".scala", ".sc"), _FULL, "scala"),
    "lua": LanguageSupport(
        "lua",
        "Lua",
        (".lua",),
        _CHUNK,
        "lua",
        frozenset({"function_declaration", "variable_declaration", "function_call"}),
    ),
    "bash": LanguageSupport(
        "bash",
        "Bash / Shell",
        (".sh", ".bash", ".zsh"),
        _CHUNK,
        "bash",
        frozenset(
            {
                "function_definition",
                "if_statement",
                "for_statement",
                "while_statement",
                "case_statement",
                "command",
            }
        ),
    ),
    "sql": LanguageSupport("sql", "SQL", (".sql",), _CHUNK, "sql", frozenset({"statement"})),
    "html": LanguageSupport(
        "html",
        "HTML",
        (".html", ".htm"),
        _STRUCTURED,
        "html",
        frozenset({"element", "script_element", "style_element"}),
    ),
    "css": LanguageSupport(
        "css",
        "CSS",
        (".css",),
        _STRUCTURED,
        "css",
        frozenset({"rule_set", "media_statement", "import_statement"}),
    ),
    "xml": LanguageSupport("xml", "XML", (".xml",), _STRUCTURED, "xml", frozenset({"element"})),
    "yaml": LanguageSupport(
        "yaml", "YAML", (".yaml", ".yml"), _STRUCTURED, "yaml", frozenset({"document"})
    ),
    "toml": LanguageSupport(
        "toml", "TOML", (".toml",), _CHUNK, "toml", frozenset({"table", "table_array_element"})
    ),
    "json": LanguageSupport(
        "json", "JSON", (".json",), _CHUNK, "json", frozenset({"object", "array"})
    ),
    "markdown": LanguageSupport(
        "markdown",
        "Markdown",
        (".md", ".markdown"),
        _STRUCTURED,
        "markdown",
        frozenset({"section"}),
    ),
    "solidity": LanguageSupport(
        "solidity",
        "Solidity",
        (".sol",),
        _CHUNK,
        "solidity",
        frozenset(
            {
                "contract_declaration",
                "interface_declaration",
                "library_declaration",
                "function_definition",
                "pragma_directive",
            }
        ),
    ),
}

EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".php": "php",
    ".rb": "ruby",
    ".scala": "scala",
    ".sc": "scala",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".md": "markdown",
    ".markdown": "markdown",
    ".sol": "solidity",
}

LANGUAGE_PACK_NAMES: dict[str, str] = {
    language_id: support.pack_name for language_id, support in LANGUAGE_SUPPORT.items()
}

CHUNK_ONLY_LANGUAGE_IDS: frozenset[str] = frozenset(
    language_id
    for language_id, support in LANGUAGE_SUPPORT.items()
    if support.tier == LanguageTier.CHUNK_ONLY
)
STRUCTURED_LANGUAGE_IDS: frozenset[str] = frozenset(
    language_id
    for language_id, support in LANGUAGE_SUPPORT.items()
    if support.tier == LanguageTier.STRUCTURED
)


def get_language_support(language_id: str) -> LanguageSupport | None:
    return LANGUAGE_SUPPORT.get(language_id)


def get_language_tier(language_id: str) -> LanguageTier:
    support = get_language_support(language_id)
    return support.tier if support is not None else LanguageTier.UNKNOWN
