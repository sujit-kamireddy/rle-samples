"""Deterministic simulation of the Contoso Cloud competitive-intelligence world.

The production agent grounds itself in three governed sources (Fabric IQ, an
OneLake-backed document store, and Work IQ) plus untrusted public web search.
Every one of those, except the web, is a small static snapshot -- see
``data/fabric-snapshot/context.json`` and ``data/workiq-mock/organization.json``.

That is what makes reinforcement learning tractable here: the snapshot carries an
``exposure_score`` and a ``materiality_threshold``, so the correct materiality
verdict is *computable* rather than a matter of taste. This module widens that
snapshot into an expandable world so we can generate thousands of tasks whose
ground truth is known exactly.

Only the four companies already present in the shipped snapshot use real vendor
names, and the sample already frames those as fictional analysis. Every company
added here is invented, so no synthetic claim is ever attributed to a real firm.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Iterable

# "Today" in the simulated world. Every generated document date is anchored to
# this so that provenance is always internally consistent.
REFERENCE_DATE = date(2026, 9, 20)

MATERIALITY_THRESHOLD = 0.65

# Strategic-alignment score above which the materiality policy treats an event as
# strategically aligned. Mirrors "strategic alignment above 0.80" in
# data/fabric-snapshot/context.json -> knowledge_documents -> Materiality Policy.
STRATEGIC_ALIGNMENT_BAR = 0.80

SEMANTIC_MODEL = "Competitive Intelligence Demo"
SEMANTIC_MODEL_ID = "af43930e-ce01-4950-873a-3b3b46696e1a"
WORKSPACE = "Fabric-Foundry"

# Anything older than this many days relative to the task reference date is stale
# and must be disclosed as such.
STALENESS_DAYS = 400


@dataclass(frozen=True)
class PortfolioEntry:
    """One governed portfolio-exposure row in the Fabric IQ semantic model."""

    company: str
    theme: str
    internal_product: str
    customer_segment: str
    exposure_score: float
    materiality_threshold: float = MATERIALITY_THRESHOLD

    @property
    def above_threshold(self) -> bool:
        return self.exposure_score >= self.materiality_threshold

    def as_row(self) -> dict[str, object]:
        return {
            "company": self.company,
            "theme": self.theme,
            "internal_product": self.internal_product,
            "customer_segment": self.customer_segment,
            "exposure_score": round(self.exposure_score, 2),
            "materiality_threshold": self.materiality_threshold,
        }


@dataclass(frozen=True)
class StrategicPriority:
    name: str
    weight: float


@dataclass(frozen=True)
class KnowledgeDoc:
    """An OneLake-backed document reachable through the Fabric IQ model."""

    title: str
    source_url: str
    summary: str
    published: str

    def as_row(self) -> dict[str, object]:
        return {
            "title": self.title,
            "source_url": self.source_url,
            "summary": self.summary,
            "published": self.published,
        }


@dataclass(frozen=True)
class Team:
    name: str
    leader: str
    focus: tuple[str, ...]


@dataclass(frozen=True)
class RoutingRule:
    themes: tuple[str, ...]
    stakeholders: tuple[str, ...]


# --------------------------------------------------------------------------
# Seed content taken verbatim from the shipped snapshot.
# --------------------------------------------------------------------------

SEED_PORTFOLIO: tuple[PortfolioEntry, ...] = (
    PortfolioEntry("AWS", "agent runtime", "Contoso Agent Runtime", "Enterprise developers", 0.91),
    PortfolioEntry(
        "Google", "agent interoperability", "Contoso Tool Gateway", "Platform engineering", 0.82
    ),
    PortfolioEntry(
        "Databricks", "agent optimization", "Contoso Optimizer", "Data and AI teams", 0.93
    ),
    PortfolioEntry(
        "Salesforce",
        "enterprise agents",
        "Contoso Business Context",
        "Sales and service organizations",
        0.76,
    ),
)

SEED_PRIORITIES: tuple[StrategicPriority, ...] = (
    StrategicPriority("Durable enterprise agents", 1.0),
    StrategicPriority("Adaptive agent quality", 0.95),
    StrategicPriority("Enterprise grounding", 0.90),
    StrategicPriority("Secure agent actions", 0.85),
)

SEED_TEAMS: tuple[Team, ...] = (
    Team(
        "Agent Platform",
        "Jordan Lee",
        ("hosted agent runtime", "agent evaluation", "tool governance"),
    ),
    Team(
        "Enterprise Data and IQ",
        "Avery Chen",
        ("semantic models", "enterprise context", "data permissions"),
    ),
    Team(
        "Field Strategy",
        "Morgan Patel",
        ("competitive positioning", "customer adoption", "industry analysis"),
    ),
)

SEED_ROUTING: tuple[RoutingRule, ...] = (
    RoutingRule(
        ("agent runtime", "agent optimization", "model platform"),
        ("Agent Platform", "Field Strategy"),
    ),
    RoutingRule(
        ("enterprise data", "semantic model", "knowledge graph"),
        ("Enterprise Data and IQ", "Agent Platform"),
    ),
    RoutingRule(
        ("agent interoperability", "tool security", "agent governance"),
        ("Agent Platform", "Enterprise Data and IQ"),
    ),
    RoutingRule(
        ("enterprise agents", "industry solutions", "customer adoption"),
        ("Field Strategy",),
    ),
)

STATIC_DOCS: tuple[KnowledgeDoc, ...] = (
    KnowledgeDoc(
        "Strategy Brief",
        "onelake://CompetitiveIntelligenceDemo/Files/competitive-knowledge/strategy-brief.md",
        (
            "# Contoso Cloud agent strategy\n\nThis document is fictional and exists only for "
            "the demo.\n\nContoso Cloud differentiates through durable hosted execution, "
            "governed access to business context, evaluation-driven optimization, and "
            "centralized tool security. Prioritize announcements that affect more than one of "
            "these pillars.\n"
        ),
        "2026-02-11",
    ),
    KnowledgeDoc(
        "Materiality Policy",
        "onelake://CompetitiveIntelligenceDemo/Files/competitive-knowledge/materiality-policy.md",
        (
            "# Competitive event materiality policy\n\nThis document is fictional and exists "
            "only for the Contoso Cloud demo.\n\nAn event is material only when **both** "
            "governed conditions hold: the company's governed exposure score is greater than "
            "or equal to its materiality threshold, **and** its strategic alignment is greater "
            "than or equal to 0.80. Both numbers are returned in the governed portfolio row; "
            "read them, do not estimate them. Loud coverage is not exposure, and a high "
            "exposure score on a deprioritized theme is still not material. Where credible "
            "evidence is absent, unresolved, or merely rumored, abstain rather than guess. "
            "Analysts must cite the primary announcement, corroborate material claims, query "
            "governed portfolio exposure, and disclose stale data.\n"
        ),
        "2026-01-23",
    ),
)


# --------------------------------------------------------------------------
# Generated (fictional) competitors used to widen the task distribution.
# --------------------------------------------------------------------------

_FICTIONAL_COMPANIES: tuple[str, ...] = (
    "Nimbus Systems",
    "Helio AI",
    "Cobalt Analytics",
    "Northwind Labs",
    "Lumen Stack",
    "Vertex Grid",
    "Quanta Forge",
    "Ironleaf Software",
    "Meridian Compute",
    "Halcyon Data",
    "Solstice AI",
    "Beacon Logic",
    "Tessellate Cloud",
    "Harbor Point Systems",
    "Kestrel Works",
    "Orbital Reason",
    "Pinnacle Semantics",
    "Redwood Signal",
    "Silverline AI",
    "Trellis Dynamics",
    "Umbra Platforms",
    "Veridian Compute",
    "Waypoint Intelligence",
    "Zenith Fabric",
    "Auric Networks",
    "Basalt AI",
    "Cirrus Reason",
    "Delta Loom",
    "Ember Gridworks",
    "Fathom Systems",
    "Glacier Model Co",
    "Hollowpeak AI",
)

# The thirty-two names above were written by hand and are the pool every dataset
# built before the single-turn recipe drew on. RL needs far more distinct tasks
# than that -- the reference tool-calling recipe sees most of its tasks exactly
# once -- so the pool is widened combinatorially below.
#
# The generated names are *appended*, never interleaved, so that
# ``_FICTIONAL_COMPANIES[:32]`` still names exactly the original pool and
# ``build_world(n_generated=32)`` still reproduces, entry for entry, the world
# every earlier run and every frozen eval set was built from. Widening the pool
# must not silently restate what a previous benchmark measured.
_GENERATED_STEMS: tuple[str, ...] = (
    "Alder", "Ansible", "Arcwright", "Ashgrove", "Aster", "Axiom", "Bellwether",
    "Blackwood", "Brightwater", "Calder", "Carbon", "Cascade", "Chandler",
    "Cinder", "Clearwater", "Copperline", "Crestwood", "Dovetail", "Driftwood",
    "Eastgate", "Elmshaw", "Fairlight", "Farthing", "Ferrous", "Flintlock",
    "Foxglove", "Gallant", "Gatehouse", "Goldleaf", "Granite", "Greenbriar",
    "Hawthorn", "Hearthstone", "Highmark", "Hollis", "Inkwell", "Ironwood",
    "Juniper", "Keystone", "Kingfisher", "Larkspur", "Lattice", "Lodestar",
    "Longview", "Marblehead", "Millrace", "Moorland", "Nightjar", "Norwood",
    "Oakhurst", "Obsidian", "Orchard", "Paxton", "Peregrine", "Pewter",
    "Quarry", "Quillon", "Ravenswood", "Rimfire", "Rooksbridge", "Saltmarsh",
    "Sandpiper", "Sheffield", "Slatebrook", "Sparrow", "Stonebridge",
    "Sundial", "Thistle", "Thornbury", "Tidewater", "Timberline", "Torchlight",
    "Underhill", "Vantage", "Verdigris", "Wellspring", "Westbrook", "Wheatley",
    "Whitlock", "Wilder", "Windlass", "Wolfram", "Yarrow", "Yewtree",
)

_GENERATED_SUFFIXES: tuple[str, ...] = (
    "Systems", "AI", "Labs", "Compute", "Analytics", "Dynamics", "Networks",
    "Platforms", "Intelligence", "Reason",
)


def _generated_companies() -> tuple[str, ...]:
    """Stem x suffix names, ordered so the pool grows without reordering.

    Suffix varies fastest within a stem, which keeps the sequence stable when a
    stem is appended: every prefix of this tuple is a prefix of every longer
    one. Names that collide with a hand-written entry are skipped rather than
    deduplicated later, because a duplicate company would silently halve that
    entry's exposure evidence.
    """
    taken = {name.casefold() for name in _FICTIONAL_COMPANIES}
    out: list[str] = []
    for stem in _GENERATED_STEMS:
        for suffix in _GENERATED_SUFFIXES:
            name = f"{stem} {suffix}"
            if name.casefold() in taken:
                continue
            taken.add(name.casefold())
            out.append(name)
    return tuple(out)


#: The full pool: the hand-written names first, then the generated ones.
ALL_COMPANIES: tuple[str, ...] = _FICTIONAL_COMPANIES + _generated_companies()

# Theme -> the Contoso product that theme overlaps with.
_THEME_PRODUCTS: tuple[tuple[str, str, str], ...] = (
    ("agent runtime", "Contoso Agent Runtime", "Enterprise developers"),
    ("agent interoperability", "Contoso Tool Gateway", "Platform engineering"),
    ("agent optimization", "Contoso Optimizer", "Data and AI teams"),
    ("enterprise agents", "Contoso Business Context", "Sales and service organizations"),
    ("agent evaluation", "Contoso Optimizer", "Data and AI teams"),
    ("tool security", "Contoso Tool Gateway", "Platform engineering"),
    ("semantic model", "Contoso Business Context", "Enterprise data teams"),
    ("model platform", "Contoso Agent Runtime", "Enterprise developers"),
    ("agent observability", "Contoso Optimizer", "Platform engineering"),
    ("agent governance", "Contoso Tool Gateway", "Risk and compliance"),
)

# Theme -> the strategic priority it maps onto, for alignment scoring.
_THEME_PRIORITY: dict[str, str] = {
    "agent runtime": "Durable enterprise agents",
    "agent interoperability": "Secure agent actions",
    "agent optimization": "Adaptive agent quality",
    "enterprise agents": "Enterprise grounding",
    "agent evaluation": "Adaptive agent quality",
    "tool security": "Secure agent actions",
    "semantic model": "Enterprise grounding",
    "model platform": "Durable enterprise agents",
    "agent observability": "Adaptive agent quality",
    "agent governance": "Secure agent actions",
}

_PRODUCT_NOUNS: tuple[str, ...] = (
    "Agent Mesh",
    "Flow Runtime",
    "Reason Studio",
    "Context Bridge",
    "Signal Optimizer",
    "Policy Gateway",
    "Insight Fabric",
    "Pipeline Agents",
    "Trust Layer",
    "Model Router",
)


@dataclass(frozen=True)
class World:
    """The complete governed world visible to the simulated tools."""

    portfolio: tuple[PortfolioEntry, ...]
    priorities: tuple[StrategicPriority, ...]
    docs: tuple[KnowledgeDoc, ...]
    teams: tuple[Team, ...]
    routing: tuple[RoutingRule, ...]
    captured_at: str = "2026-09-14T13:51:57.9081505Z"

    _by_company: dict[str, PortfolioEntry] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_by_company", {e.company.casefold(): e for e in self.portfolio}
        )

    def semantic_model_name(self) -> str:
        return SEMANTIC_MODEL

    def semantic_model_identifier(self) -> str:
        return SEMANTIC_MODEL_ID

    def lookup(self, company: str | None) -> PortfolioEntry | None:
        if not company:
            return None
        return self._by_company.get(company.casefold())

    def priority_weight(self, theme: str) -> float:
        name = _THEME_PRIORITY.get(theme)
        for priority in self.priorities:
            if priority.name == name:
                return priority.weight
        return 0.0

    def stakeholders_for(self, theme: str) -> tuple[str, ...]:
        for rule in self.routing:
            if theme in rule.themes:
                return rule.stakeholders
        return ("Field Strategy",)

    def dossier_for(self, company: str) -> KnowledgeDoc | None:
        wanted = f"competitor {company}".casefold()
        for doc in self.docs:
            if doc.title.casefold() == wanted:
                return doc
        return None


def _dossier(company: str, theme: str, product: str, published: str) -> KnowledgeDoc:
    slug = company.lower().replace(" ", "-")
    return KnowledgeDoc(
        title=f"Competitor {company}",
        source_url=(
            f"onelake://CompetitiveIntelligenceDemo/Files/competitive-knowledge/"
            f"competitor-{slug}.md"
        ),
        summary=(
            f"# {company} competitive dossier\n\nThis fictional analysis is based on public "
            f"product information. The company's {theme} work is relevant to {product} because "
            f"it can shift customer expectations in that category.\n"
        ),
        published=published,
    )


def _stable_float(*parts: str) -> float:
    """Deterministic float in [0, 1) derived from the given strings."""
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def build_world(
    *,
    n_generated: int = len(_FICTIONAL_COMPANIES),
    seed: int = 7,
) -> World:
    """Construct the full world: the shipped snapshot plus generated competitors.

    Exposure scores are spread deliberately across the 0.65 threshold. The shipped
    snapshot only contains material companies (0.76-0.93); training exclusively on
    that would teach the model to answer "material" unconditionally.
    """
    rng = random.Random(seed)
    portfolio: list[PortfolioEntry] = list(SEED_PORTFOLIO)
    docs: list[KnowledgeDoc] = list(STATIC_DOCS)

    # Dossiers for the two seeded competitors that ship with one.
    docs.append(
        _dossier("Google", "agent interoperability", "Contoso Tool Gateway", "2026-03-04")
    )
    docs.append(
        _dossier("Databricks", "agent optimization", "Contoso Optimizer", "2026-04-18")
    )

    chosen = list(ALL_COMPANIES[:n_generated])
    for index, company in enumerate(chosen):
        theme, product, segment = _THEME_PRODUCTS[index % len(_THEME_PRODUCTS)]

        # Alternate deliberately around the threshold so both verdicts are common,
        # and place roughly a fifth of entries within +/-0.06 of the boundary to
        # force the model to actually read the number instead of guessing.
        bucket = index % 5
        base = _stable_float(company, theme)
        if bucket in (0, 1):
            exposure = 0.68 + base * 0.30            # clearly material
        elif bucket in (2, 3):
            exposure = 0.18 + base * 0.42            # clearly not material
        else:
            exposure = MATERIALITY_THRESHOLD - 0.06 + base * 0.12  # boundary case
        exposure = round(min(0.99, max(0.05, exposure)), 2)

        portfolio.append(PortfolioEntry(company, theme, product, segment, exposure))

        # Most, but not all, competitors have a dossier. A missing dossier is a
        # legitimate evidence gap the model must disclose rather than invent.
        #
        # The date is anchored to REFERENCE_DATE and always lands in the past,
        # comfortably inside STALENESS_DAYS. Deriving it from the loop index
        # instead let dossiers fall after the announcements they analysed, and a
        # correctly calibrated agent reads an impossible publication date as a
        # provenance defect and abstains - on tasks whose label says MATERIAL.
        # That penalised exactly the caution the other variants reward.
        if rng.random() < 0.75:
            days_ago = 30 + int(_stable_float(company, theme, "dossier") * 270)
            published = (REFERENCE_DATE - timedelta(days=days_ago)).isoformat()
            docs.append(_dossier(company, theme, product, published))

    return World(
        portfolio=tuple(portfolio),
        priorities=SEED_PRIORITIES,
        docs=tuple(docs),
        teams=SEED_TEAMS,
        routing=SEED_ROUTING,
    )


def product_name(company: str, index: int = 0) -> str:
    """Deterministic fictional product name for a company's announcement."""
    noun = _PRODUCT_NOUNS[
        int(_stable_float(company, str(index)) * len(_PRODUCT_NOUNS)) % len(_PRODUCT_NOUNS)
    ]
    first = company.split()[0]
    return f"{first} {noun}"


def split_companies(
    world: World,
    *,
    holdout_fraction: float = 0.22,
    seed: int = 13,
    protect: Iterable[str] = (),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition companies into train and validation pools.

    The split is over *entities*, not queries. A validation company is never seen
    during training, so the eval curve measures generalization to new competitors
    rather than recall of memorized exposure scores.

    ``protect`` names companies that must land in the holdout pool whatever the
    shuffle does. Callers pass the golden set's companies: keying the exclusion on
    ``SEED_PORTFOLIO`` alone used to miss the fictional competitors that golden
    draws on, which let "Nimbus Systems" into training while golden scored the
    model on abstaining from a Nimbus rumour.
    """
    companies = [entry.company for entry in world.portfolio]
    rng = random.Random(seed)
    shuffled = list(companies)
    rng.shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * holdout_fraction))
    holdout = shuffled[:n_holdout]

    # The four snapshot companies drive the frozen golden eval, so keep them out of
    # the training pool to avoid contaminating that comparison.
    seeded = {entry.company for entry in SEED_PORTFOLIO}
    holdout_set = set(holdout) | seeded | set(protect)
    train = tuple(c for c in companies if c not in holdout_set)
    validation = tuple(c for c in companies if c in holdout_set)
    return train, validation


def visible_portfolio(world: World, companies: Iterable[str]) -> tuple[PortfolioEntry, ...]:
    wanted = {c.casefold() for c in companies}
    return tuple(e for e in world.portfolio if e.company.casefold() in wanted)


def with_exposure(entry: PortfolioEntry, exposure: float) -> PortfolioEntry:
    return replace(entry, exposure_score=round(exposure, 2))
