Generate a chart specification for Jira issue data visualization.

Return ONLY a JSON object — no markdown, no explanation, no code fences:
{"type": "<type>", "x_field": "<field>", "y_field": "<field>", "title": "<title>", "color_field": <"field" or null>}

Rules:
- type: if the user names a chart type use it exactly. Infer only when not specified:
  "stacked_bar" when two grouping dimensions are needed (e.g. assignee + status),
  "bar" for counts by a single category, "pie" for proportions,
  "line" for trends over time, "scatter" for correlations.
  Aliases: "multi-line" → "line", "area" → "line".
- x_field: primary grouping — use exact field names only:
  "assignee", "status", "issuetype", "priority", "sprint",
  "created", "updated", "reporter", "labels"
- y_field: "count" to count issues, or a numeric field name (e.g. "story_points") to sum values
- color_field: secondary grouping for stacked_bar (e.g. "status" stacked on "assignee"); null for all other types
- title: short human-readable chart title

User chart request: {chart_hint}
