"""The production tool surface, served over the synthetic world.

Why this module exists
----------------------
``simulated_tools`` exposes four invented semantic sources --
``fabric_iq_query``, ``onelake_knowledge_search``, ``org_context_lookup`` and
``work_iq_search``. Production exposes none of them. A ``tools/list`` against
the live toolbox returns:

    competitive_fabric___DiscoverArtifacts(searchQuery, artifactTypes, maxResults)
    competitive_fabric___ExecuteQuery(artifactId, daxQueries, maxRows)
    competitive_fabric___GetReportMetadata(reportObjectId, queries)
    competitive_fabric___GetSemanticModelSchema(artifactId, queries)
    competitive_fabric___ResolveReportIdFromUrl(url)
    competitive_fabric___ValueSearch(artifactId, searchTerms, scope)
    web_search(search_query)

Only ``web_search`` is common to both, so a policy trained on the invented
surface meets six unseen tools at evaluation time. That shows up exactly where
you would expect: the synthetic split scores tool discipline at 0.97 while the
production benchmark scores the same checkpoint at 0.43, and the observed
failure is a hallucinated ``artifactId`` -- a GUID the policy had never had to
obtain, because no tool it trained against took one.

The original surface was chosen deliberately, on the argument that "training a
policy to emit brittle DAX strings teaches string formatting, not competitive
analysis". The production measurement inverts that argument: obtaining a valid
artifact id *is* the thing the policy fails at, and it fails at it because the
environment removed it.

What this module does and does not change
-----------------------------------------
Facts come from the same per-task ``World`` as the legacy surfaces. The bounded
DAX interpreter applies actual filters, projections and aggregates, rejecting
unsupported syntax explicitly. Evidence credit follows returned cell provenance,
not merely a table mentioned in a query. The facts have to be reached the way
production reaches them: discover an artifact, carry its id, and query it.

The ids are per-task and derived from the world, so they cannot be memorised
across tasks and cannot be guessed -- they can only be discovered. Calls that
invent one get the real toolbox's own rejection, verbatim in shape:

    {"Answer": "...", "Status": "error",
     "Error": {"Code": "InvalidArgument", "Source": "PowerBIWorkload",
               "HttpStatusCode": 400, "Message": "...",
               "IsRetryable": false, "IsUserError": true}}

Those rejections are returned, never raised, which matches both the hosted
rollout path and the production dispatch loop in ``toolbox.py``: an argument the
model can fix costs it a turn, not the investigation.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Any

from loom_cookbook.tool_use import ToolResult, simple_tool_result

from .dax import SUPPORTED, DaxError, evaluate_query

try:  # pragma: no cover - exercised only by the packaged environment
    from loom_cookbook.tool_use import tool
except ImportError:  # pragma: no cover
    from loom_cookbook.tool_use.tools import tool  # type: ignore[no-redef]

#: The shape the toolbox accepts. Anything else is ``InvalidArgument`` before a
#: lookup is attempted, which is what makes "I made up a GUID" a distinct,
#: correctable failure rather than a generic miss.
GUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

#: Tool names on the production surface, for grading and for the decoy check.
PRODUCTION_READ_ONLY_TOOL_NAMES = frozenset(
    {
        "web_search",
        "competitive_fabric___DiscoverArtifacts",
        "competitive_fabric___GetSemanticModelSchema",
        "competitive_fabric___ExecuteQuery",
        "competitive_fabric___ValueSearch",
        "competitive_fabric___GetReportMetadata",
        "competitive_fabric___ResolveReportIdFromUrl",
    }
)

#: Tables the synthetic semantic model presents. A DAX query has to name one of
#: these, which is what ``GetSemanticModelSchema`` is for.
EXPOSURE_TABLE = "PortfolioExposure"
KNOWLEDGE_TABLE = "KnowledgeDocs"
STAKEHOLDER_TABLE = "Stakeholders"
TABLE_COLUMNS = {
    EXPOSURE_TABLE: (
        "company", "theme", "internal_product", "customer_segment", "exposure_score",
        "materiality_threshold", "strategic_alignment", "reporting_period", "unit",
    ),
    KNOWLEDGE_TABLE: ("title", "source_url", "summary", "published"),
    STAKEHOLDER_TABLE: ("name", "leader", "focus"),
}


def _guid_for(*parts: str) -> str:
    """A stable, well-formed v4-shaped GUID for a synthetic artifact.

    Derived from the world rather than randomly assigned so a rollout is
    reproducible, and derived from the *task* rather than fixed so the id cannot
    be memorised across tasks. Discovery is the only way to learn it.
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return (
        f"{digest[0:8]}-{digest[8:12]}-4{digest[13:16]}-"
        f"8{digest[17:20]}-{digest[20:32]}"
    )


def _error_result(
    message: str,
    *,
    code: str = "InvalidArgument",
    source: str = "PowerBIWorkload",
    status: int = 400,
) -> ToolResult:
    """The toolbox's error envelope, reproduced field for field.

    ``IsUserError``/``IsRetryable`` are the fields the production dispatch loop
    keys off to decide whether a failure is correctable, so they have to be
    present and honest here or the two paths diverge again.
    """
    return simple_tool_result(
        json.dumps(
            {
                "Answer": message,
                "Status": "error",
                "Error": {
                    "Code": code,
                    "Source": source,
                    "HttpStatusCode": status,
                    "Message": message,
                    "IsRetryable": False,
                    "IsUserError": True,
                },
                "DurationMs": 1,
            },
            indent=2,
        )
    )


class ProductionToolsMixin:
    """The seven production tools, mixed into :class:`ToolSession`.

    A mixin rather than a subclass so there is exactly one session type and one
    construction site: the environment picks a *surface*, not a class, and all
    the per-episode bookkeeping the graders read stays on one object.
    """

    # ------------------------------------------------------------------
    # Artifact catalogue
    # ------------------------------------------------------------------

    def _reject(
        self,
        message: str,
        *,
        code: str = "InvalidArgument",
        source: str = "PowerBIWorkload",
        status: int = 400,
    ) -> ToolResult:
        """Return a correctable rejection and count it.

        The count is what lets tool discipline distinguish "called ExecuteQuery"
        from "got data out of ExecuteQuery". A policy that invents an artifact id
        still produces a call record; only the rejection counter shows the call
        bought nothing.
        """
        self.rejected_calls = getattr(self, "rejected_calls", 0) + 1
        return _error_result(message, code=code, source=source, status=status)

    def _semantic_model_guid(self) -> str:
        return _guid_for("semantic-model", self.world.semantic_model_name())

    def _report_guid(self) -> str:
        company = self.task.company or "portfolio"
        return _guid_for("report", company, self.world.semantic_model_name())

    def _catalogue(self) -> dict[str, dict[str, Any]]:
        """Every artifact this task can reach, keyed by id."""
        model_id = self._semantic_model_guid()
        report_id = self._report_guid()
        company = self.task.company or "Portfolio"
        return {
            model_id: {
                "Name": self.world.semantic_model_name(),
                "ArtifactId": model_id,
                "ArtifactType": "Semantic Model",
                "DirectLink": (
                    f"https://msit.powerbi.com/datahub/datasets/{model_id}"
                ),
                "WorkspaceName": "BICOE_Prod_CompetitiveIntelligence",
                "Description": (
                    "Governed portfolio exposure for tracked competitors, with "
                    "the materiality threshold and strategic alignment weight "
                    "each assessment is judged against."
                ),
                "Endorsement": "Certified",
                "IsPreparedForCopilot": True,
            },
            report_id: {
                "Name": f"{company} Competitive Threat Brief",
                "ArtifactId": report_id,
                "ArtifactType": "Report",
                "DirectLink": (
                    f"https://msit.powerbi.com/groups/me/reports/{report_id}"
                ),
                "WorkspaceName": "BICOE_Prod_CompetitiveIntelligence",
                "Description": (
                    "Published brief summarising exposure and recent signals for "
                    "the tracked competitor."
                ),
                "Endorsement": "Promoted",
                "IsPreparedForCopilot": True,
            },
        }

    def _resolve_artifact(
        self, artifact_id: Any, *, parameter: str = "ArtifactId"
    ) -> tuple[dict[str, Any] | None, ToolResult | None]:
        """Validate then look up an artifact id.

        Returns ``(artifact, None)`` on success and ``(None, error)`` otherwise.
        The two failures are kept distinct on purpose: a malformed id and an
        absent one are different mistakes and carry different corrections.
        """
        value = "" if artifact_id is None else str(artifact_id).strip()
        if not GUID_PATTERN.match(value):
            return None, self._reject(
                f"The '{parameter}' parameter must be a valid non-empty GUID. "
                f"Received: '{value}'."
            )
        artifact = self._catalogue().get(value.lower()) or self._catalogue().get(value)
        if artifact is None:
            return None, self._reject(
                "Power BI couldn't find the artifact. It may have been moved, "
                "deleted, or you may not have access.",
                code="ArtifactNotFound",
                status=404,
            )
        return artifact, None

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool
    async def competitive_fabric___DiscoverArtifacts(  # noqa: N802
        self,
        searchQuery: Annotated[  # noqa: N803
            str,
            "The text to search for artifacts (Semantic Models and Reports). "
            "Must be a non-empty search term.",
        ],
        artifactTypes: Annotated[  # noqa: N803
            list[str] | None,
            "Optional filter for artifact types. Supported values: "
            "'SemanticModel', 'Report'.",
        ] = None,
        maxResults: Annotated[  # noqa: N803
            int | None, "Maximum number of results to return (default: 5, max: 50)."
        ] = None,
    ) -> ToolResult:
        """Search for Power BI artifacts (Semantic Models and Reports).

        Returns each match's name, artifact ID, artifact type, direct link and
        workspace. Call this first: the artifact ID it returns is what every
        other Fabric tool needs.
        """
        query = (searchQuery or "").strip()
        if not query:
            self._record(
                "competitive_fabric___DiscoverArtifacts",
                {"searchQuery": searchQuery},
                [],
            )
            return self._reject(
                "The 'searchQuery' parameter is required and cannot be null, "
                "empty, or whitespace. Provide a non-empty search term "
                "describing the artifacts to look for.",
                source="FabricAIHub",
            )

        wanted: set[str] | None = None
        if artifactTypes:
            wanted = {str(t).replace(" ", "").casefold() for t in artifactTypes}

        artifacts = []
        for artifact in self._catalogue().values():
            if wanted is not None:
                if artifact["ArtifactType"].replace(" ", "").casefold() not in wanted:
                    continue
            artifacts.append(artifact)

        limit = maxResults if isinstance(maxResults, int) and maxResults > 0 else 5
        artifacts = artifacts[: min(limit, 50)]
        payload = {
            "searchQuery": query,
            "totalResults": len(artifacts),
            "artifacts": artifacts,
        }
        self._record(
            "competitive_fabric___DiscoverArtifacts",
            {"searchQuery": query},
            artifacts,
        )
        return simple_tool_result(json.dumps(payload, indent=2))

    @tool
    async def competitive_fabric___GetSemanticModelSchema(  # noqa: N802
        self,
        artifactId: Annotated[  # noqa: N803
            str, "The GUID of the artifact to fetch the schema for."
        ],
        queries: Annotated[
            list[str] | None,
            "Optional JMESPath expressions to query specific parts of the schema.",
        ] = None,
    ) -> ToolResult:
        """Retrieve the tables, columns and measures of a semantic model.

        Use the schema to write a DAX query that names real tables and columns.
        """
        artifact, error = self._resolve_artifact(artifactId)
        if error is not None:
            self._record(
                "competitive_fabric___GetSemanticModelSchema",
                {"artifactId": artifactId},
                [],
            )
            return error

        schema = {
            "artifactId": artifact["ArtifactId"],
            "name": artifact["Name"],
            "tables": [
                {
                    "name": name,
                    "columns": list(columns),
                    "measures": [],
                }
                for name, columns in TABLE_COLUMNS.items()
            ],
            "customInstructions": (
                "exposure_score is a normalized index on 0-1. Compare it against "
                "materiality_threshold from the same row; do not compare it "
                "against a threshold from memory. An empty result means no rows "
                "match your filter, not zero exposure or no impact. This is a "
                "simulated model, not live organizational data. Supported DAX: "
                f"{SUPPORTED}. Other constructs return UnsupportedDaxQuery."
            ),
            "capturedAt": self.world.captured_at,
        }
        self._record(
            "competitive_fabric___GetSemanticModelSchema",
            {"artifactId": artifactId},
            schema["tables"],
        )
        return simple_tool_result(json.dumps(schema, indent=2))

    @tool
    async def competitive_fabric___ExecuteQuery(  # noqa: N802
        self,
        artifactId: Annotated[  # noqa: N803
            str, "The GUID of the artifact to execute the DAX query against."
        ],
        daxQueries: Annotated[  # noqa: N803
            list[str],
            "The DAX queries to execute (min: 1, max: 4). Each query must start "
            "with EVALUATE.",
        ],
        maxRows: Annotated[  # noqa: N803
            int | None, "The maximum number of rows to return per query."
        ] = None,
    ) -> ToolResult:
        """Execute a DAX query against a semantic model and return the results.

        Governed exposure lives here. Query PortfolioExposure for the exposure
        score and the materiality threshold it must be judged against.
        """
        artifact, error = self._resolve_artifact(artifactId)
        if error is not None:
            self._record(
                "competitive_fabric___ExecuteQuery",
                {"artifactId": artifactId, "daxQueries": daxQueries},
                [],
            )
            return error

        if (
            not isinstance(daxQueries, list)
            or not 1 <= len(daxQueries) <= 4
            or any(not isinstance(query, str) or not query.strip() for query in daxQueries)
        ):
            self._record(
                "competitive_fabric___ExecuteQuery",
                {"artifactId": artifactId, "daxQueries": daxQueries},
                [],
            )
            return self._reject(
                "The 'daxQueries' parameter requires between 1 and 4 queries."
            )
        if maxRows is not None and (type(maxRows) is not int or maxRows < 1):
            self._record(
                "competitive_fabric___ExecuteQuery",
                {"artifactId": artifactId, "daxQueries": daxQueries, "maxRows": maxRows},
                [],
            )
            return self._reject(
                "The 'maxRows' parameter must be a positive integer."
            )

        queries = daxQueries
        tables = self._dax_tables()
        delivered = set()
        results = []
        for dax in queries:
            try:
                result = evaluate_query(dax, tables, TABLE_COLUMNS, maxRows or 250)
            except DaxError as error:
                self._record(
                    "competitive_fabric___ExecuteQuery",
                    {"artifactId": artifactId, "daxQueries": queries},
                    [],
                )
                return self._reject(str(error), code=error.code)
            delivered.update(result.sources)
            results.append({
                "query": dax,
                "rowCount": len(result.rows),
                "rows": result.rows,
                "truncated": result.total_rows > len(result.rows),
            })

        # No credit until the whole response succeeds, including maxRows.
        for table, index, column in delivered:
            row = tables[table][index]
            if table == EXPOSURE_TABLE:
                if column in ("exposure_score", "materiality_threshold"):
                    self.served_numbers.add(f"{row[column]:.2f}")
                if column == "exposure_score" and row["company"] == self.task.company:
                    self.served_governed = True
            elif table == KNOWLEDGE_TABLE and column == "source_url":
                self.served_doc_urls.add(row[column])
            elif table == STAKEHOLDER_TABLE and column == "name":
                self.served_stakeholders.add(row[column])
        payload = {
            "artifactId": artifact["ArtifactId"],
            "semanticModel": artifact["Name"],
            "capturedAt": self.world.captured_at,
            "results": results,
        }
        self._record(
            "competitive_fabric___ExecuteQuery",
            {"artifactId": artifactId, "daxQueries": queries},
            [row for r in results for row in r["rows"]],
        )
        return simple_tool_result(json.dumps(payload, indent=2))

    def _dax_tables(self) -> dict[str, list[dict[str, Any]]]:
        """Pure model snapshot; materializing it does not deliver any evidence."""
        return {
            EXPOSURE_TABLE: [
                row for entry in self.world.portfolio
                for row in self._governed_payload(entry.company, record=False)["rows"]
            ],
            KNOWLEDGE_TABLE: [doc.as_row() for doc in self.world.docs],
            STAKEHOLDER_TABLE: [
                {"name": team.name, "leader": team.leader, "focus": "; ".join(team.focus)}
                for team in self.world.teams
            ],
        }

    @tool
    async def competitive_fabric___ValueSearch(  # noqa: N802
        self,
        artifactId: Annotated[  # noqa: N803
            str, "The GUID of the artifact to search values in."
        ],
        searchTerms: Annotated[  # noqa: N803
            list[str], "The values to search for in the semantic model (1 to 20)."
        ],
        scope: Annotated[
            list[str] | None,
            "Optional scope restricting the search to specific tables or columns.",
        ] = None,
    ) -> ToolResult:
        """Resolve entity names to exact table/column/value locations.

        Call this before ExecuteQuery when the question names a specific company,
        so the DAX filter uses the value as the model actually spells it.
        """
        artifact, error = self._resolve_artifact(artifactId)
        if error is not None:
            self._record(
                "competitive_fabric___ValueSearch",
                {"artifactId": artifactId, "searchTerms": searchTerms},
                [],
            )
            return error

        terms = [str(t).strip() for t in (searchTerms or []) if str(t).strip()]
        if not terms:
            return self._reject(
                "The 'searchTerms' parameter requires between 1 and 20 values."
            )

        matches = []
        for term in terms[:20]:
            entry = self.world.lookup(term)
            if entry is not None:
                matches.append(
                    {
                        "searchTerm": term,
                        "table": EXPOSURE_TABLE,
                        "column": "company",
                        "value": entry.company,
                        "exactMatch": entry.company.casefold() == term.casefold(),
                    }
                )
            else:
                matches.append({"searchTerm": term, "matches": []})

        payload = {
            "artifactId": artifact["ArtifactId"],
            "results": matches,
            "note": (
                "A term with no match is not in the governed portfolio. Exposure "
                "for it is unknown, not zero."
            ),
        }
        self._record(
            "competitive_fabric___ValueSearch",
            {"artifactId": artifactId, "searchTerms": terms},
            matches,
        )
        return simple_tool_result(json.dumps(payload, indent=2))

    @tool
    async def competitive_fabric___GetReportMetadata(  # noqa: N802
        self,
        reportObjectId: Annotated[  # noqa: N803
            str, "The object ID of the Power BI report to analyze."
        ],
        queries: Annotated[
            list[str] | None,
            "Optional JMESPath expressions to query parts of the report metadata.",
        ] = None,
    ) -> ToolResult:
        """Retrieve a report's workspace, semantic model, pages and visuals."""
        artifact, error = self._resolve_artifact(
            reportObjectId, parameter="ReportObjectId"
        )
        if error is not None:
            self._record(
                "competitive_fabric___GetReportMetadata",
                {"reportObjectId": reportObjectId},
                [],
            )
            return error
        if artifact["ArtifactType"] != "Report":
            return self._reject(
                f"The artifact '{artifact['ArtifactId']}' is a "
                f"{artifact['ArtifactType']}, not a Report. Use "
                "GetSemanticModelSchema for semantic models."
            )

        metadata = {
            "reportObjectId": artifact["ArtifactId"],
            "name": artifact["Name"],
            "workspaceName": artifact["WorkspaceName"],
            "semanticModelId": self._semantic_model_guid(),
            "pages": [
                {
                    "name": "Exposure Overview",
                    "visuals": ["Exposure by competitor", "Threshold reference line"],
                },
                {"name": "Recent Signals", "visuals": ["Signal timeline"]},
            ],
            "capturedAt": self.world.captured_at,
        }
        self._record(
            "competitive_fabric___GetReportMetadata",
            {"reportObjectId": reportObjectId},
            metadata["pages"],
        )
        return simple_tool_result(json.dumps(metadata, indent=2))

    @tool
    async def competitive_fabric___ResolveReportIdFromUrl(  # noqa: N802
        self,
        url: Annotated[str, "A Power BI report URL."],
    ) -> ToolResult:
        """Resolve a Power BI report URL to its report ID."""
        found = re.search(
            r"/reports/([0-9a-fA-F-]{36})", str(url or "")
        ) or re.search(r"([0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12})", str(url or ""))
        if not found:
            self._record(
                "competitive_fabric___ResolveReportIdFromUrl", {"url": url}, []
            )
            return self._reject(
                "The 'url' parameter must be a Power BI report URL containing a "
                f"report id. Received: '{url}'."
            )
        report_id = found.group(1)
        artifact, error = self._resolve_artifact(
            report_id, parameter="ReportObjectId"
        )
        if error is not None:
            self._record(
                "competitive_fabric___ResolveReportIdFromUrl", {"url": url}, []
            )
            return error
        self._record(
            "competitive_fabric___ResolveReportIdFromUrl",
            {"url": url},
            [artifact["ArtifactId"]],
        )
        return simple_tool_result(
            json.dumps(
                {"url": url, "reportObjectId": artifact["ArtifactId"]}, indent=2
            )
        )
