

/* 1. Issues updated within the last exact 60 minutes - live pipeline ingestion trigger */
updated >= -60m ORDER BY updated DESC

/* 2. Issues created in the last 30 minutes - real-time alerting feed for new tickets */
created >= -30m ORDER BY created DESC

/* 3. Issues whose due date is strictly in the past right now - overdue at this exact moment */
project = HADOOP AND duedate < now() AND resolution = Unresolved ORDER BY duedate ASC

/* 4. Issues due within the next 4 hours - ultra-short SLA window for ops teams */
project = KAFKA AND duedate >= now() AND duedate <= "4h" AND resolution = Unresolved ORDER BY duedate ASC

/* 5. Blocker bugs not touched in the last 2 hours - stale critical issues right now */
issuetype = Bug AND priority = Blocker AND updated <= -2h AND resolution = Unresolved ORDER BY updated ASC

/* 6. Issues created within the last 5 minutes - used as a streaming changelog cursor */
created >= -5m ORDER BY created DESC

/* 7. Issues resolved in the last 10 minutes - near-real-time closure feed */
resolved >= -10m AND resolution IS NOT EMPTY ORDER BY resolved DESC

/* 8. Issues whose status changed within the last 15 minutes - workflow event stream */
status CHANGED AFTER -15m ORDER BY updated DESC

/* 9. Issues created but not yet resolved that are past their due date right now */
duedate < now() AND created < now() AND resolution = Unresolved ORDER BY duedate ASC

/* 10. Issues created in the last 1 hour that are already Blocker priority - hot new fires */
created >= -1h AND priority = Blocker ORDER BY created DESC



/* 11. Issues created today from midnight onwards - daily ingestion baseline */
project = SPARK AND created >= startOfDay() ORDER BY created DESC

/* 12. Issues resolved today - end-of-day closure count for dashboards */
resolved >= startOfDay() AND resolution = Fixed ORDER BY resolved DESC

/* 13. Issues updated today but created before today - revived old issues */
updated >= startOfDay() AND created < startOfDay() AND resolution = Unresolved ORDER BY updated DESC

/* 14. Issues created yesterday - previous business day report (startOfDay with -1d offset) */
created >= startOfDay("-1d") AND created < startOfDay() ORDER BY created ASC

/* 15. Issues created the day before yesterday - two-day lag report */
created >= startOfDay("-2d") AND created < startOfDay("-1d") ORDER BY created ASC

/* 16. Issues due today (between midnight and end of day) - daily task list */
project = HADOOP AND duedate >= startOfDay() AND duedate <= endOfDay() AND resolution = Unresolved ORDER BY priority ASC

/* 17. Issues due tomorrow - one-day advance warning */
duedate >= startOfDay("1d") AND duedate <= endOfDay("1d") AND resolution = Unresolved ORDER BY project ASC, priority ASC

/* 18. Issues due in the next 3 days (today through end of day+2) - 3-day lookahead */
duedate >= startOfDay() AND duedate <= endOfDay("2d") AND resolution = Unresolved ORDER BY duedate ASC

/* 19. Issues past due as of end of yesterday - officially overdue as of today morning */
duedate < startOfDay() AND resolution = Unresolved ORDER BY duedate ASC

/* 20. Priority changed to Blocker today - escalation events in current business day */
priority CHANGED TO Blocker AFTER startOfDay() ORDER BY updated DESC

/* 21. Status moved to Resolved today - same-day resolution rate */
status CHANGED TO Resolved AFTER startOfDay() AND resolution = Fixed ORDER BY updated DESC

/* 22. Issues created today with no comment yet - new issues needing first response */
created >= startOfDay() AND comment IS EMPTY AND resolution = Unresolved ORDER BY priority ASC, created ASC

/* 23. Issues whose due date was changed today (due date slippage tracking) */
duedate CHANGED AFTER startOfDay() ORDER BY updated DESC

/* 24. Issues assigned today (assignee field changed today) - today's triage assignments */
assignee CHANGED AFTER startOfDay() ORDER BY updated DESC

/* 25. Fix version assigned for the first time today (version planning activity today) */
fixVersion CHANGED FROM EMPTY AFTER startOfDay() ORDER BY project ASC, updated DESC




/* 26. Issues created since the start of the current week - weekly intake report */
created >= startOfWeek() ORDER BY created DESC

/* 27. Issues resolved this week - weekly throughput / velocity metric */
resolved >= startOfWeek() AND resolution IS NOT EMPTY ORDER BY resolved DESC

/* 28. Issues created last week using -1w offset - previous week intake comparison */
created >= startOfWeek("-1w") AND created < startOfWeek() ORDER BY created ASC

/* 29. Issues resolved last week - previous week resolution comparison */
resolved >= startOfWeek("-1w") AND resolved < startOfWeek() AND resolution IS NOT EMPTY ORDER BY resolved ASC

/* 30. Issues created 2 weeks ago - two-week-ago cohort for trend analysis */
created >= startOfWeek("-2w") AND created < startOfWeek("-1w") ORDER BY created ASC

/* 31. Issues due by end of this week - this-week deadline list */
duedate >= startOfDay() AND duedate <= endOfWeek() AND resolution = Unresolved ORDER BY duedate ASC

/* 32. Issues due between end of this week and end of next week - next-week planning horizon */
duedate > endOfWeek() AND duedate <= endOfWeek("1w") AND resolution = Unresolved ORDER BY duedate ASC

/* 33. Blocker bugs created this week across Apache - weekly Blocker intake count */
issuetype = Bug AND priority = Blocker AND created >= startOfWeek() ORDER BY project ASC, created ASC

/* 34. Issues whose status changed to In Progress this week - weekly WIP starts */
status CHANGED TO "In Progress" AFTER startOfWeek() ORDER BY project ASC, updated DESC

/* 35. Issues whose assignee changed this week - weekly reassignment activity */
assignee CHANGED AFTER startOfWeek() ORDER BY updated DESC

/* 36. Issues created on Monday this week - using startOfWeek() which anchors to Monday (EU) */
created >= startOfWeek() AND created < startOfWeek("1d") ORDER BY created ASC

/* 37. Issues created on exactly the last three Mondays - rolling Monday cohort */
(created >= startOfWeek("-1w") AND created < startOfWeek("-1w") )
OR (created >= startOfWeek("-8d") AND created < startOfWeek("-7d"))
OR (created >= startOfWeek("-15d") AND created < startOfWeek("-14d"))
ORDER BY created DESC

/* 38. Issues updated this week but created before this year - long-tail revival this week */
updated >= startOfWeek() AND created < startOfYear() AND resolution = Unresolved ORDER BY created ASC

/* 39. Issues moved to Patch Available this week (weekly patch submission rate) */
status CHANGED TO "Patch Available" AFTER startOfWeek() ORDER BY project ASC, updated DESC

/* 40. Security issues that had any field changed this week - weekly security change audit */
labels = "security" AND updated >= startOfWeek() ORDER BY updated DESC



/* 41. Issues created this calendar month - monthly intake total */
created >= startOfMonth() ORDER BY created DESC

/* 42. Issues resolved this calendar month - monthly throughput total */
resolved >= startOfMonth() AND resolution IS NOT EMPTY ORDER BY resolved DESC

/* 43. Issues created last month - using startOfMonth(-1) for previous month */
created >= startOfMonth("-1") AND created < startOfMonth() ORDER BY created ASC

/* 44. Issues resolved last month - previous month resolution total for reporting */
resolved >= startOfMonth("-1") AND resolved < startOfMonth() AND resolution IS NOT EMPTY ORDER BY resolved ASC

/* 45. Issues created 2 months ago - two-month-ago cohort for trend comparison */
created >= startOfMonth("-2") AND created < startOfMonth("-1") ORDER BY created ASC

/* 46. Issues due this month that are not yet resolved - monthly deadline board */
duedate >= startOfMonth() AND duedate <= endOfMonth() AND resolution = Unresolved ORDER BY duedate ASC, priority ASC

/* 47. Issues due next month - forward-looking monthly planning list */
duedate >= startOfMonth("1") AND duedate <= endOfMonth("1") AND resolution = Unresolved ORDER BY project ASC, duedate ASC

/* 48. Issues whose due date passed this month without resolution - monthly SLA failures */
duedate >= startOfMonth() AND duedate < now() AND resolution = Unresolved ORDER BY duedate ASC

/* 49. Blocker bugs created this month per project - monthly Blocker rate report */
issuetype = Bug AND priority = Blocker AND created >= startOfMonth() ORDER BY project ASC, created ASC

/* 50. Issues resolved as Fixed this month - monthly fixed-bug count */
resolution = Fixed AND resolved >= startOfMonth() ORDER BY project ASC, resolved DESC

/* 51. Issues created this month that still have no assignee - monthly triage gap */
created >= startOfMonth() AND assignee IS EMPTY AND resolution = Unresolved ORDER BY priority ASC, created ASC

/* 52. Issues where fix version changed this month - monthly version planning shuffles */
fixVersion CHANGED AFTER startOfMonth() AND resolution = Unresolved ORDER BY updated DESC

/* 53. New features resolved this month - monthly shipped features count */
issuetype = "New Feature" AND resolution = Fixed AND resolved >= startOfMonth() ORDER BY project ASC

/* 54. Issues created in Q1 of the current year (Jan–Mar) */
created >= startOfYear() AND created <= endOfMonth("2") ORDER BY created ASC

/* 55. Issues created in Q2 of the current year (Apr–Jun) */
created >= startOfMonth("3") AND created <= endOfMonth("5") AND created >= startOfYear() ORDER BY created ASC

/* 56. Issues created in Q3 of the current year (Jul–Sep) */
created >= startOfMonth("6") AND created <= endOfMonth("8") AND created >= startOfYear() ORDER BY created ASC

/* 57. Issues created in Q4 of the current year (Oct–Dec) */
created >= startOfMonth("9") AND created <= endOfYear() ORDER BY created ASC

/* 58. Issues updated in the first 7 days of the current month - month-start activity burst */
updated >= startOfMonth() AND updated < startOfMonth("7d") AND resolution = Unresolved ORDER BY updated DESC



/* 59. Issues created this calendar year - year-to-date intake total */
created >= startOfYear() ORDER BY project ASC, created DESC

/* 60. Issues resolved this calendar year - year-to-date throughput */
resolved >= startOfYear() AND resolution IS NOT EMPTY ORDER BY resolved DESC

/* 61. Issues created last year - previous year cohort for annual comparison */
created >= startOfYear("-1") AND created < startOfYear() ORDER BY created ASC

/* 62. Issues still open that were created last year - last-year unresolved backlog debt */
created >= startOfYear("-1") AND created < startOfYear() AND resolution = Unresolved ORDER BY priority ASC, created ASC

/* 63. Blocker bugs created this year resolved as Won't Fix - triage quality analysis */
issuetype = Bug AND priority = Blocker AND created >= startOfYear() AND resolution = "Won't Fix" ORDER BY resolved DESC

/* 64. Issues resolved in under 1 day this year - same-day fix rate (fast resolution SLA) */
resolution = Fixed AND resolved >= startOfYear() AND created >= -365d ORDER BY created DESC

/* 65. Issues created at the start of this year (first 7 days) - new-year backlog seeding */
created >= startOfYear() AND created < startOfYear("7d") ORDER BY created ASC

/* 66. Issues that were Blocker at the very start of this year (snapshot query) */
priority WAS Blocker ON startOfYear() ORDER BY project ASC

/* 67. Issues that were Open at the very start of this year (annual backlog snapshot) */
status WAS Open ON startOfYear() ORDER BY project ASC, priority ASC

/* 68. Issues resolved in the last quarter of last year - year-end sprint closure */
resolved >= startOfMonth("-14") AND resolved <= endOfYear("-1") AND resolution IS NOT EMPTY ORDER BY resolved DESC

/* 69. Issues created this year in Apache security projects - annual security intake */
project IN (HADOOP, KAFKA, TOMCAT, RANGER, KNOX) AND labels = "security" AND created >= startOfYear() ORDER BY project ASC, priority ASC

/* 70. Issues that have been unresolved since before this year - carry-over backlog from prior year */
created < startOfYear() AND resolution = Unresolved ORDER BY project ASC, created ASC



/* 71. Issues created in the last 7 days AND resolved within the same 7-day window - fast-close rate */
created >= -7d AND resolved >= -7d AND resolution = Fixed ORDER BY resolved DESC

/* 72. Issues created more than 90 days ago but updated within the last 7 days - ancient-revived */
created <= -90d AND updated >= -7d AND resolution = Unresolved ORDER BY created ASC

/* 73. Issues created more than 365 days ago with no update in 180 days - permanent backlog debt */
created <= -365d AND updated <= -180d AND resolution = Unresolved ORDER BY updated ASC

/* 74. Issues created in the last 14 days that are already In Patch Available - fast patch turnaround */
created >= -14d AND status = "Patch Available" AND resolution = Unresolved ORDER BY created ASC

/* 75. Issues where priority changed within 24h of creation - triage speed indicator */
priority CHANGED AFTER -24h AND created >= -2d ORDER BY updated DESC

/* 76. Issues that took more than 30 days from creation to first Patch Available - slow patch rate */
status WAS "Patch Available" AND created <= -30d AND resolution = Unresolved ORDER BY created ASC

/* 77. Issues unresolved for between 7 and 30 days - active aging window */
created >= -30d AND created <= -7d AND resolution = Unresolved ORDER BY project ASC, created ASC

/* 78. Issues unresolved for between 30 and 90 days - mature aging window */
created >= -90d AND created <= -30d AND resolution = Unresolved ORDER BY project ASC, created ASC

/* 79. Issues unresolved for between 90 and 180 days - chronic aging window */
created >= -180d AND created <= -90d AND resolution = Unresolved ORDER BY project ASC, created ASC

/* 80. Issues unresolved for between 180 days and 1 year - deep backlog window */
created >= -365d AND created <= -180d AND resolution = Unresolved ORDER BY project ASC, created ASC

/* 81. Issues resolved in exactly the same day they were created (resolved within 24h of creation) */
resolution = Fixed AND resolved >= created AND resolved <= startOfDay("1d") AND created >= -365d ORDER BY created DESC

/* 82. Issues updated between 7 and 14 days ago - mid-cycle activity audit */
updated >= -14d AND updated <= -7d AND resolution = Unresolved ORDER BY project ASC, updated ASC

/* 83. Issues created in the last 48 hours that have already had a comment - fast community response */
created >= -2d AND comment IS NOT EMPTY ORDER BY comment DESC

/* 84. Issues where due date is between 7 and 14 days from now - medium-term deadline radar */
duedate >= startOfDay("7d") AND duedate <= endOfDay("14d") AND resolution = Unresolved ORDER BY duedate ASC

/* 85. Issues with no activity for exactly 2 years (between 730 and 731 days old, never updated) */
created <= -730d AND created >= -731d AND created = updated AND resolution = Unresolved ORDER BY project ASC



/* 86. Issues created in a specific fiscal quarter using hard-coded absolute dates */
created >= "2025-01-01" AND created <= "2025-03-31" ORDER BY project ASC, created ASC

/* 87. Issues created in a specific calendar month using absolute date boundaries */
created >= "2025-11-01" AND created < "2025-12-01" ORDER BY project ASC, created ASC

/* 88. Issues created on a specific day (hour precision: midnight to midnight) */
created >= "2025-06-15 00:00" AND created < "2025-06-16 00:00" ORDER BY project ASC, created ASC

/* 89. Issues resolved in a defined past quarter using absolute date range */
resolved >= "2024-10-01" AND resolved <= "2024-12-31" AND resolution IS NOT EMPTY ORDER BY project ASC, resolved DESC

/* 90. Issues created since a major release date AND updated within the last 30 days - post-release activity */
created >= "2025-05-23" AND updated >= -30d AND resolution = Unresolved ORDER BY updated DESC

/* 91. Issues created in the last 30 days but with a due date hard-coded to a past sprint end - overdue sprint work */
created >= -30d AND duedate < "2025-03-31" AND resolution = Unresolved ORDER BY duedate ASC

/* 92. Issues updated after a specific infrastructure migration date - post-migration issue surge */
updated >= "2025-04-01" AND created < "2025-04-01" AND resolution = Unresolved ORDER BY updated DESC

/* 93. Issues that were In Progress on the last day of the previous year (historical snapshot) */
status WAS "In Progress" ON "2024-12-31" ORDER BY project ASC

/* 94. Issues that were Blocker priority on a specific quarterly review date */
priority WAS Blocker ON "2025-09-30" ORDER BY project ASC

/* 95. Issues created in last 90 days but resolved by a specific hard-coded deadline */
created >= -90d AND resolved <= "2025-12-31" AND resolution = Fixed ORDER BY resolved DESC



/* 96. Issues whose status changed to Resolved AFTER the start of last month AND BEFORE the start of this month */
status CHANGED TO Resolved AFTER startOfMonth("-1") AND status CHANGED TO Resolved BEFORE startOfMonth() ORDER BY project ASC, updated DESC

/* 97. Issues whose priority was escalated to Blocker within the last 7 days AND that are still unresolved */
priority CHANGED TO Blocker AFTER -7d AND resolution = Unresolved ORDER BY project ASC, created ASC

/* 98. Issues whose assignee changed in the current week AND whose fix version is in the next unreleased version - release re-assignments */
assignee CHANGED AFTER startOfWeek() AND fixVersion in unreleasedVersions() AND resolution = Unresolved ORDER BY updated DESC

/* 99. Issues whose fix version was deferred (changed) this month AND that are Blocker or Critical - release risk tracker */
fixVersion CHANGED AFTER startOfMonth() AND priority IN (Blocker, Critical) AND resolution = Unresolved ORDER BY priority ASC, updated DESC

/* 100. Multi-field time intelligence: Blocker bugs created this year, updated this week, with Patch Available status and a due date before end of this month */
issuetype = Bug AND priority = Blocker AND created >= startOfYear() AND updated >= startOfWeek() AND status = "Patch Available" AND duedate <= endOfMonth() AND resolution = Unresolved ORDER BY duedate ASC, created ASC

/* 101. Find issues in epics that still have unresolved stories */
issueFunction in issuesInEpics("project = PROJ AND resolution IS EMPTY")

/* 102. Return all epics that contain at least one high-priority bug */
issueFunction in epicsOf("issuetype = Bug AND priority in (High, Highest)")

/* 103. Get parent issues of subtasks assigned to current user */
issueFunction in parentsOf("issuetype = Sub-task AND assignee = currentUser()")

/* 104. All subtasks whose parent is in status 'In Progress' */
issueFunction in subtasksOf("status = 'In Progress'")

/* 105. Issues whose worklog time exceeds original estimate by 50% */
issueFunction in expression("", "timespent > originalestimate * 1.5")

/* 106. Issues whose remaining estimate is less than 1 hour */
issueFunction in expression("", "remainingEstimate < 3600")

/* 107. Issues commented on by current user in the last 3 days */
issueFunction in commented("by currentUser() after -3d")

/* 108. Issues commented on more than 5 times */
issueFunction in commented("> 5")

/* 109. Issues that have at least one subtask */
issueFunction in hasSubtasks()

/* 110. Issues that have at least one linked issue */
issueFunction in hasLinks("")

/* 111. Issues in epics targeted for release-1 */
issueFunction in issuesInEpics("labels = release-1")

/* 112. Epics that contain at least one blocked story */
issueFunction in epicsOf("issueFunction in linkedIssuesOf('issuetype = Story', 'is blocked by')")

/* 113. Parents of subtasks updated in the last 24 hours */
issueFunction in parentsOf("issuetype = Sub-task AND updated >= -1d")

/* 114. Subtasks of high-priority issues in backlog */
issueFunction in subtasksOf("project = PROJ AND priority in (High, Highest) AND sprint IS EMPTY")

/* 115. Issues where logged time is zero but estimate exists */
issueFunction in expression("", "originalestimate > 0 AND timespent = 0")

/* 116. Issues where logged time exceeds 2x the original estimate */
issueFunction in expression("", "timespent >= originalestimate * 2")

/* 117. Issues you commented on but do not own */
issueFunction in commented("by currentUser()") AND assignee != currentUser()

/* 118. Issues commented on by 'ops-team' user in last week */
issueFunction in commented("by ops-team after -7d")

/* 119. Issues that have at least one attachment (ScriptRunner-aware) */
issueFunction in hasAttachments()

/* 120. Issues that have at least one inward or outward link of any type */
issueFunction in hasLinks("any")

/* 121. All tests inside test executions matching a filter (e.g., Xray) */
issueFunction in testsOf("project = TEST AND issuetype = 'Test Execution'")

/* 122. Test executions that contain at least one failed test */
issueFunction in testExecutionsOf("status = Failed")

/* 123. Issues that match a regex on summary via issueFieldMatch */
issueFunction in issueFieldMatch("project = PROJ", "summary", ".*API.*v2.*")

/* 124. Issues whose description contains an incident ID pattern */
issueFunction in issueFieldMatch("project = OPS", "description", "INC-[0-9]{4}")

/* 125. Issues where a custom field matches a complex numeric expression */
issueFunction in issueFieldMatch("project = PROJ", "Story Points", "^[89]|1[0-3]$")



/* issueFunction not in <functionName>("<subquery>") */

/* 126. Issues NOT in any epic that has unresolved stories */
issueFunction not in epicsOf("resolution IS EMPTY")

/* 127. Issues whose epics do NOT belong to project PORTFOLIO */
issueFunction not in issuesInEpics("project = PORTFOLIO")

/* 128. Issues whose subtasks are NOT assigned to current user */
issueFunction not in parentsOf("assignee = currentUser()")

/* 129. Stories that are NOT subtasks of any In Progress parent */
issuetype = Story AND issueFunction not in subtasksOf("status = 'In Progress'")

/* 130. Issues that do NOT violate estimate vs time spent rule */
issueFunction not in expression("", "timespent > originalestimate * 1.5")

/* 131. Issues that do NOT have remaining estimate below 1 hour */
issueFunction not in expression("", "remainingEstimate < 3600")

/* 132. Issues current user has never commented on */
issueFunction not in commented("by currentUser()")

/* 133. Issues that have fewer than 2 comments */
issueFunction not in commented(">= 2")

/* 134. Issues that do NOT have any subtasks */
issueFunction not in hasSubtasks()

/* 135. Issues with absolutely no links (inverting hasLinks) */
issueFunction not in hasLinks("")

/* 136. Issues NOT part of any epic tagged 'release-1' */
issueFunction not in issuesInEpics("labels = release-1")

/* 137. Epics that do NOT have any blocked stories */
issueFunction not in epicsOf("issueFunction in linkedIssuesOf('issuetype = Story', 'is blocked by')")

/* 138. Parents that do NOT have any subtasks updated recently */
issueFunction not in parentsOf("updated >= -2d")

/* 139. Issues that are NOT subtasks of high-priority items */
issueFunction not in subtasksOf("priority in (High, Highest)")

/* 140. Issues with estimate but NOT zero logged time */
issueFunction not in expression("", "originalestimate > 0 AND timespent = 0")

/* 141. Issues that have NOT exceeded 2x original estimate */
issueFunction not in expression("", "timespent >= originalestimate * 2")

/* 142. Issues that current user never commented on in last 7 days */
issueFunction not in commented("by currentUser() after -7d")

/* 143. Issues with no comments from 'ops-team' in last week */
issueFunction not in commented("by ops-team after -7d")

/* 144. Issues that do NOT have attachments */
issueFunction not in hasAttachments()

/* 145. Issues that do NOT have any links of type 'blocks' or 'is blocked by' */
issueFunction not in hasLinks("blocks, is blocked by")

/* 146. Tests NOT part of any failed test execution */
issueFunction not in testsOf("status = Failed")

/* 147. Test executions that do NOT contain any failed tests */
issueFunction not in testExecutionsOf("status = Failed")

/* 148. Issues whose summary does NOT match a given regex */
issueFunction not in issueFieldMatch("project = PROJ", "summary", ".*API.*v2.*")

/* 149. Ops issues whose description does NOT contain incident IDs */
issueFunction not in issueFieldMatch("project = OPS", "description", "INC-[0-9]{4}")

/* 150. Issues whose Story Points do NOT fall in the complex range */
issueFunction not in issueFieldMatch("project = PROJ", "Story Points", "^[89]|1[0-3]$")



/* issueFunction in linkedIssuesOfRecursive("subquery") */

/* 151. All issues directly/indirectly related to a single critical issue */
issueFunction in linkedIssuesOfRecursive("issue = PROJ-1000")

/* 152. Full dependency graph for all open issues in project PROJ */
issueFunction in linkedIssuesOfRecursive("project = PROJ AND status != Done")

/* 153. All issues in the dependency tree of epics in release-2 */
issueFunction in linkedIssuesOfRecursive("issuetype = Epic AND fixVersion = 'release-2'")

/* 154. All items related to production incidents in last 7 days */
issueFunction in linkedIssuesOfRecursive("project = OPS AND issuetype = Incident AND created >= -7d")

/* 155. Entire graph of work connected to a specific feature epic */
issueFunction in linkedIssuesOfRecursive("issuetype = Epic AND key = FEAT-42")

/* 156. All issues linked directly or indirectly to security bugs */
issueFunction in linkedIssuesOfRecursive("labels = security AND issuetype = Bug")

/* 157. All items involved in the chain around a regressed bug */
issueFunction in linkedIssuesOfRecursive("summary ~ 'regression' AND issuetype = Bug")

/* 158. Complete network of issues tied to a given customer */
issueFunction in linkedIssuesOfRecursive("\"Customer\" = 'Acme Corp'")

/* 159. All issues connected to unresolved problems (ITSM) */
issueFunction in linkedIssuesOfRecursive("issuetype = Problem AND resolution IS EMPTY")

/* 160. All work linked to stories in current sprints */
issueFunction in linkedIssuesOfRecursive("issuetype = Story AND sprint in openSprints()")

/* 161. All issues linked to release blockers in release-3 */
issueFunction in linkedIssuesOfRecursive("labels = 'release-3-blocker'")

/* 162. Entire graph around high-priority bugs in backlog */
issueFunction in linkedIssuesOfRecursive("issuetype = Bug AND priority in (High, Highest) AND sprint IS EMPTY")

/* 163. All issues related to items updated today */
issueFunction in linkedIssuesOfRecursive("updated >= startOfDay()")

/* 164. All linked issues for tickets with SLA breaches */
issueFunction in linkedIssuesOfRecursive("\"Time to resolution\" = breached()")

/* 165. Dependency graph for tests that failed last run */
issueFunction in linkedIssuesOfRecursive("issuetype = Test AND status = Failed")

/* 166. All work connected to a specific component (API) */
issueFunction in linkedIssuesOfRecursive("component = API")

/* 167. Complete chain around issues with label 'migration' */
issueFunction in linkedIssuesOfRecursive("labels = migration")

/* 168. All issues connected to infrastructure tasks in project INFRA */
issueFunction in linkedIssuesOfRecursive("project = INFRA")

/* 169. All dependencies touching issues created this week */
issueFunction in linkedIssuesOfRecursive("created >= startOfWeek()")

/* 170. Entire network of bugs in Production environment */
issueFunction in linkedIssuesOfRecursive("\"Environment\" = 'Production' AND issuetype = Bug")

/* 171. All issues connected to any epic with status 'In Progress' */
issueFunction in linkedIssuesOfRecursive("issuetype = Epic AND status = 'In Progress'")

/* 172. All linked issues for tickets that changed status today */
issueFunction in linkedIssuesOfRecursive("status CHANGED AFTER startOfDay()")

/* 173. All dependent tasks for features targeted at 'Mobile' component */
issueFunction in linkedIssuesOfRecursive("issuetype = Story AND component = Mobile")

/* 174. Full graph around tickets tagged 'incident-review' */
issueFunction in linkedIssuesOfRecursive("labels = 'incident-review'")

/* 175. All items tied to issues in project BACKEND that are not done */
issueFunction in linkedIssuesOfRecursive("project = BACKEND AND resolution IS EMPTY")



/* issueFunction in linkedIssuesOfRecursive("subquery", "link type") */

/* 176. All issues blocked (directly/indirectly) by PROJ-123 */
issueFunction in linkedIssuesOfRecursive("issue = PROJ-123", "is blocked by")

/* 177. All issues that PROJ-123 itself blocks, recursively */
issueFunction in linkedIssuesOfRecursive("issue = PROJ-123", "blocks")

/* 178. All issues related by 'relates to' to current sprint stories */
issueFunction in linkedIssuesOfRecursive("sprint in openSprints() AND issuetype = Story", "relates to")

/* 179. Dependency chain of bugs caused by a given problem (ITSM) */
issueFunction in linkedIssuesOfRecursive("issuetype = Problem AND key = PROB-1", "is caused by")

/* 180. All issues that duplicate or are duplicated by PROJ-50 */
issueFunction in linkedIssuesOfRecursive("issue = PROJ-50", "duplicates")

/* 181. All issues blocked (recursively) by high-priority bugs */
issueFunction in linkedIssuesOfRecursive("issuetype = Bug AND priority in (High, Highest)", "is blocked by")

/* 182. All issues that a set of epics 'depend on' for delivery */
issueFunction in linkedIssuesOfRecursive("issuetype = Epic AND fixVersion = 'release-4'", "depends on")

/* 183. All items that 'clones' link connects to legacy tickets */
issueFunction in linkedIssuesOfRecursive("project = LEGACY", "clones")

/* 184. Cross-project dependencies where FRONTEND depends on BACKEND */
issueFunction in linkedIssuesOfRecursive("project = FRONTEND", "depends on")

/* 185. All downstream issues blocked by production incidents */
issueFunction in linkedIssuesOfRecursive("\"Environment\" = 'Production' AND issuetype = Incident", "blocks")

/* 186. All tickets that are 'caused by' a failed deployment issue */
issueFunction in linkedIssuesOfRecursive("labels = 'failed-deploy'", "is caused by")

/* 187. All issues that 'implements' a given design ticket */
issueFunction in linkedIssuesOfRecursive("component = Design", "implements")

/* 188. Backend tasks that 'relate to' mobile stories */
issueFunction in linkedIssuesOfRecursive("project = MOBILE AND issuetype = Story", "relates to")

/* 189. Issues that 'blocks' chain for release-blocker bugs */
issueFunction in linkedIssuesOfRecursive("labels = 'release-blocker'", "blocks")

/* 190. Items that 'is blocked by' chain for API gateway epic */
issueFunction in linkedIssuesOfRecursive("key = API-EPIC-1", "is blocked by")

/* 191. All issues 'clones' linked to spike investigations */
issueFunction in linkedIssuesOfRecursive("issuetype = Spike", "clones")

/* 192. Work that 'depends on' infrastructure tasks in INFRA */
issueFunction in linkedIssuesOfRecursive("project = INFRA", "depends on")

/* 193. All tickets 'relates to' security audit items */
issueFunction in linkedIssuesOfRecursive("labels = 'security-audit'", "relates to")

/* 194. Stories that 'implements' high-level architecture tickets */
issueFunction in linkedIssuesOfRecursive("issuetype = 'Architecture'", "implements")

/* 195. All bugs 'is caused by' configuration issues */
issueFunction in linkedIssuesOfRecursive("labels = 'config-issue'", "is caused by")

/* 196. All downstream items 'depends on' database migration tasks */
issueFunction in linkedIssuesOfRecursive("labels = 'db-migration'", "depends on")

/* 197. All upstream items that 'blocks' UX redesign stories */
issueFunction in linkedIssuesOfRecursive("component = UX AND issuetype = Story", "blocks")

/* 198. Tickets 'relates to' A/B test experiments */
issueFunction in linkedIssuesOfRecursive("labels = 'ab-test'", "relates to")

/* 199. All tasks 'implements' experimental feature flags */
issueFunction in linkedIssuesOfRecursive("labels = 'feature-flag'", "implements")

/* 200. All items 'is blocked by' technical debt tasks */
issueFunction in linkedIssuesOfRecursive("labels = 'tech-debt'", "is blocked by")

/* 201. All issues 'depends on' vendor integration tasks */
issueFunction in linkedIssuesOfRecursive("labels = 'vendor-integration'", "depends on")



/* issueFunction in linkedIssuesOfRecursive("subquery", "link type 1", "link type 2") */

/* 202. Issues blocked by or blocking PROJ-500 (both directions) */
issueFunction in linkedIssuesOfRecursive("issue = PROJ-500", "blocks", "is blocked by")

/* 203. All items that either 'relates to' or 'duplicates' legacy tickets */
issueFunction in linkedIssuesOfRecursive("project = LEGACY", "relates to", "duplicates")

/* 204. Dependency network for epics that 'depends on' or 'is caused by' others */
issueFunction in linkedIssuesOfRecursive("issuetype = Epic", "depends on", "is caused by")

/* 205. All issues tied to incidents via 'blocks' or 'is caused by' links */
issueFunction in linkedIssuesOfRecursive("issuetype = Incident", "blocks", "is caused by")

/* 206. All work connected to security bugs via 'relates to' or 'blocks' */
issueFunction in linkedIssuesOfRecursive("labels = security AND issuetype = Bug", "relates to", "blocks")

/* 207. Items that 'clones' or 'duplicates' tickets in project BACKLOG */
issueFunction in linkedIssuesOfRecursive("project = BACKLOG", "clones", "duplicates")

/* 208. Full chain of items that 'depends on' or 'blocks' DB migration tasks */
issueFunction in linkedIssuesOfRecursive("labels = 'db-migration'", "depends on", "blocks")

/* 209. All issues 'relates to' or 'implements' design stories */
issueFunction in linkedIssuesOfRecursive("component = Design AND issuetype = Story", "relates to", "implements")

/* 210. Cross-project dependencies via 'depends on' and 'relates to' links */
issueFunction in linkedIssuesOfRecursive("project in (FRONTEND, BACKEND)", "depends on", "relates to")

/* 211. Items linked to production incidents by 'blocks' or 'is blocked by' */
issueFunction in linkedIssuesOfRecursive("\"Environment\" = 'Production' AND issuetype = Incident", "blocks", "is blocked by")

/* 212. All work tied to release-blocker issues via 'relates to' or 'blocks' */
issueFunction in linkedIssuesOfRecursive("labels = 'release-blocker'", "relates to", "blocks")

/* 213. Tickets involved in 'duplicates' or 'is caused by' chains from config issues */
issueFunction in linkedIssuesOfRecursive("labels = 'config-issue'", "duplicates", "is caused by")

/* 214. Items related to 'ab-test' experiments via 'relates to' and 'blocks' */
issueFunction in linkedIssuesOfRecursive("labels = 'ab-test'", "relates to", "blocks")

/* 215. Issues that 'implements' or 'depends on' architecture decisions */
issueFunction in linkedIssuesOfRecursive("issuetype = 'Architecture'", "implements", "depends on")

/* 216. Work tied to vendor integration via 'relates to' or 'depends on' links */
issueFunction in linkedIssuesOfRecursive("labels = 'vendor-integration'", "relates to", "depends on")

/* 217. All tasks connected to feature flags via 'implements' and 'relates to' */
issueFunction in linkedIssuesOfRecursive("labels = 'feature-flag'", "implements", "relates to")

/* 218. All items connected to 'tech-debt' via 'blocks' or 'relates to' */
issueFunction in linkedIssuesOfRecursive("labels = 'tech-debt'", "blocks", "relates to")

/* 219. Issues linked to 'incident-review' tickets via 'relates to' and 'is caused by' */
issueFunction in linkedIssuesOfRecursive("labels = 'incident-review'", "relates to", "is caused by")

/* 220. All issues that 'clones' or 'relates to' UX redesign stories */
issueFunction in linkedIssuesOfRecursive("component = UX AND issuetype = Story", "clones", "relates to")

/* 221. Items participating in both 'depends on' and 'is blocked by' chains from INFRA */
issueFunction in linkedIssuesOfRecursive("project = INFRA", "depends on", "is blocked by")

/* 222. Full network of incidents and problems via 'is caused by' and 'relates to' */
issueFunction in linkedIssuesOfRecursive("issuetype in (Incident, Problem)", "is caused by", "relates to")

/* 223. Issues linked to tests via 'blocks' or 'is blocked by' */
issueFunction in linkedIssuesOfRecursive("issuetype = Test", "blocks", "is blocked by")

/* 224. All items connected to customer 'Acme Corp' via 'relates to' and 'is blocked by' */
issueFunction in linkedIssuesOfRecursive("\"Customer\" = 'Acme Corp'", "relates to", "is blocked by")

/* 225. Complete chain around performance issues via 'relates to' and 'depends on' */
issueFunction in linkedIssuesOfRecursive("labels = 'performance'", "relates to", "depends on")



/* issueFunction + time/date combinations - dependency functions with time-aware subqueries */

/* 226. Epics containing at least one story created today - today's epic intake view */
issueFunction in epicsOf("created >= startOfDay() AND issuetype = Story AND resolution = Unresolved") ORDER BY updated DESC

/* 227. All issues inside epics that were updated this week - weekly epic activity scope */
issueFunction in issuesInEpics("updated >= startOfWeek() AND resolution = Unresolved") ORDER BY updated DESC

/* 228. Parent issues of subtasks created in the last 24 hours - today's subtask-spawning parents */
issueFunction in parentsOf("issuetype = Sub-task AND created >= -1d") ORDER BY priority ASC, updated DESC

/* 229. All subtasks whose parent was updated today - today's active parent work */
issueFunction in subtasksOf("updated >= startOfDay() AND resolution = Unresolved") ORDER BY priority ASC

/* 230. Issues commented on by current user today - today's engagement footprint */
issueFunction in commented("by currentUser() after startOfDay()") AND resolution = Unresolved ORDER BY updated DESC

/* 231. Issues current user commented on this week - weekly participation log */
issueFunction in commented("by currentUser() after startOfWeek()") ORDER BY updated DESC

/* 232. Issues current user commented on this month - monthly engagement audit */
issueFunction in commented("by currentUser() after startOfMonth()") ORDER BY updated DESC

/* 233. Epics with at least one resolved story this week - weekly epic completion signal */
issueFunction in epicsOf("resolved >= startOfWeek() AND resolution IS NOT EMPTY") ORDER BY updated DESC

/* 234. All issues recursively blocked by something created today - today's new blockers ripple */
issueFunction in linkedIssuesOfRecursive("created >= startOfDay() AND resolution = Unresolved", "blocks") ORDER BY priority ASC

/* 235. All issues recursively blocked by something updated this week - weekly blocker propagation */
issueFunction in linkedIssuesOfRecursive("updated >= startOfWeek() AND resolution = Unresolved", "is blocked by") ORDER BY priority ASC, updated DESC

/* 236. Parents of subtasks that were resolved today - epics/stories closed out today */
issueFunction in parentsOf("resolved >= startOfDay() AND issuetype = Sub-task") ORDER BY updated DESC

/* 237. All subtasks of issues that changed status today - today's workflow churn at subtask level */
issueFunction in subtasksOf("status CHANGED AFTER startOfDay() AND resolution = Unresolved") ORDER BY updated DESC

/* 238. All issues in epics targeted for next month (fixVersion due next month) that are Blocker */
issueFunction in issuesInEpics("duedate >= startOfMonth(\"1\") AND duedate <= endOfMonth(\"1\")") AND priority = Blocker AND resolution = Unresolved ORDER BY duedate ASC

/* 239. Epics that had their fix version reassigned this month - monthly release scope drift */
issueFunction in epicsOf("fixVersion CHANGED AFTER startOfMonth() AND resolution = Unresolved") ORDER BY updated DESC

/* 240. Fully recursive blocker chain for issues with due dates expiring today */
issueFunction in linkedIssuesOfRecursive("duedate <= endOfDay() AND resolution = Unresolved", "blocks") ORDER BY duedate ASC, priority ASC

/* 241. Parent issues of subtasks that moved to Done today - today's completion rollups */
issueFunction in parentsOf("status CHANGED TO Done AFTER startOfDay()") ORDER BY updated DESC

/* 242. Subtasks under parents whose due date has passed without resolution - overdue subtask work */
issueFunction in subtasksOf("duedate < now() AND resolution = Unresolved") AND resolution = Unresolved ORDER BY duedate ASC

/* 243. Issues with more than 3 comments, created in the last 7 days - actively discussed new issues */
issueFunction in commented("> 3") AND created >= -7d ORDER BY updated DESC

/* 244. Epics whose fix version was escalated to Blocker priority this month */
issueFunction in epicsOf("priority CHANGED TO Blocker AFTER startOfMonth() AND resolution = Unresolved") ORDER BY updated DESC

/* 245. Recursive blocker+blocked-by chain starting from issues whose priority escalated this week */
issueFunction in linkedIssuesOfRecursive("priority CHANGED TO Blocker AFTER startOfWeek()", "is blocked by", "blocks") ORDER BY priority ASC, updated DESC



/* Sprint functions combined with time - open, closed, future sprints */

/* 246. Unresolved issues in open sprints created this week - new sprint intake this week */
sprint in openSprints() AND created >= startOfWeek() AND resolution = Unresolved ORDER BY priority ASC, created DESC

/* 247. Issues in open sprints updated in the last 24 hours - today's sprint activity */
sprint in openSprints() AND updated >= -1d AND resolution = Unresolved ORDER BY updated DESC

/* 248. Issues in open sprints with due date expiring today - today's sprint deadline pressure */
sprint in openSprints() AND duedate <= endOfDay() AND resolution = Unresolved ORDER BY duedate ASC, priority ASC

/* 249. Issues in closed sprints that were resolved this month - closed-sprint monthly throughput */
sprint in closedSprints() AND resolved >= startOfMonth() AND resolution IS NOT EMPTY ORDER BY resolved DESC

/* 250. Blocker issues already assigned to future sprints - forward risk register */
sprint in futureSprints() AND priority = Blocker AND resolution = Unresolved ORDER BY project ASC, duedate ASC

/* 251. Status changed to In Progress today within open sprints - today's sprint starts */
sprint in openSprints() AND status CHANGED TO "In Progress" AFTER startOfDay() ORDER BY project ASC, updated DESC

/* 252. Issues in open sprints recursively blocking other work - sprint-level blocker propagation */
issueFunction in linkedIssuesOfRecursive("sprint in openSprints() AND resolution = Unresolved", "blocks") ORDER BY priority ASC

/* 253. Issues in open sprints with overdue due dates (past now) - sprint overdue list */
sprint in openSprints() AND resolution = Unresolved AND duedate < now() ORDER BY duedate ASC, priority ASC

/* 254. Issues left unresolved in closed sprints - sprint carry-over debt */
sprint in closedSprints() AND resolution = Unresolved ORDER BY project ASC, priority ASC

/* 255. Blocker or Critical issues added to open sprints in the last 7 days - late-sprint priority injection */
sprint in openSprints() AND created >= -7d AND priority in (Blocker, Critical) AND resolution = Unresolved ORDER BY priority ASC, created DESC

/* 256. Issues in future sprints with no assignee yet - unplanned future sprint work */
sprint in futureSprints() AND assignee IS EMPTY AND resolution = Unresolved ORDER BY project ASC, priority ASC

/* 257. Issues in open sprints with no comment and created this week - uncommented new sprint work */
sprint in openSprints() AND comment IS EMPTY AND created >= startOfWeek() AND resolution = Unresolved ORDER BY priority ASC, created ASC

/* 258. Issues moved into open sprints from closed sprints (updated after previous sprint closed) */
sprint in openSprints() AND sprint in closedSprints() AND updated >= startOfWeek() AND resolution = Unresolved ORDER BY updated DESC

/* 259. Bugs added to open sprints today - same-day sprint bug injection */
sprint in openSprints() AND issuetype = Bug AND created >= startOfDay() ORDER BY priority ASC, created DESC

/* 260. Issues in open sprints that have subtasks, still unresolved - complex open sprint work */
issueFunction in hasSubtasks() AND sprint in openSprints() AND resolution = Unresolved ORDER BY priority ASC, updated DESC



/* Version functions combined with time/date */

/* 261. Unresolved issues in unreleased versions with due dates this month - release month deadline view */
fixVersion in unreleasedVersions() AND duedate >= startOfMonth() AND duedate <= endOfMonth() AND resolution = Unresolved ORDER BY duedate ASC, priority ASC

/* 262. Issues resolved this month that were targeting released versions - shipped this month */
fixVersion in releasedVersions() AND resolved >= startOfMonth() AND resolution IS NOT EMPTY ORDER BY project ASC, resolved DESC

/* 263. Blocker issues in unreleased versions created this week - new blockers threatening release */
fixVersion in unreleasedVersions() AND priority = Blocker AND created >= startOfWeek() AND resolution = Unresolved ORDER BY project ASC, created DESC

/* 264. All unresolved issues targeting the earliest unreleased version - next-release risk list */
fixVersion = earliestUnreleasedVersion() AND resolution = Unresolved ORDER BY priority ASC, duedate ASC

/* 265. Issues created last month that are tied to the latest released version - post-release regressions */
fixVersion = latestReleasedVersion() AND created >= startOfMonth("-1") AND resolution = Unresolved ORDER BY priority ASC, created ASC

/* 266. Issues in unreleased versions updated today but still unresolved - today's release-work activity */
fixVersion in unreleasedVersions() AND updated >= startOfDay() AND resolution = Unresolved ORDER BY priority ASC, updated DESC

/* 267. Bugs resolved this year that targeted already-released versions - annual bug-fix delivery */
fixVersion in releasedVersions() AND issuetype = Bug AND resolved >= startOfYear() AND resolution = Fixed ORDER BY project ASC, resolved DESC

/* 268. Overdue issues in unreleased versions (past due, not resolved) - version SLA failures */
fixVersion in unreleasedVersions() AND duedate < now() AND resolution = Unresolved ORDER BY duedate ASC, priority ASC

/* 269. Recursive blocker chain tied to the earliest unreleased version release risk */
fixVersion = earliestUnreleasedVersion() AND issueFunction in linkedIssuesOfRecursive("labels = 'release-blocker'", "blocks") AND resolution = Unresolved ORDER BY priority ASC

/* 270. Issues in both open sprints and unreleased versions - actively worked release scope */
fixVersion in unreleasedVersions() AND sprint in openSprints() AND status = "In Progress" ORDER BY priority ASC, duedate ASC



/* CHANGED BY / WAS IN / WAS NOT IN with time boundaries */

/* 271. Issues that were In Progress at the start of today - today's carry-in WIP snapshot */
status WAS "In Progress" ON startOfDay() ORDER BY project ASC, priority ASC

/* 272. Issues that were Open at the start of this week - week's opening backlog snapshot */
status WAS Open ON startOfWeek() ORDER BY project ASC, priority ASC

/* 273. Issues that were Blocker priority at the start of this month - month-open Blocker snapshot */
priority WAS Blocker ON startOfMonth() ORDER BY project ASC

/* 274. Issues not yet Done at the start of today, still unresolved - today's inherited open work */
status WAS NOT Done ON startOfDay() AND resolution = Unresolved ORDER BY priority ASC, created ASC

/* 275. Issues that were In Progress or In Review at the start of today - today's active WIP snapshot */
status WAS IN ("In Progress", "In Review") ON startOfDay() ORDER BY project ASC, priority ASC

/* 276. Issues that were assigned to current user at the start of this week - week-opening workload */
assignee WAS currentUser() ON startOfWeek() ORDER BY priority ASC, updated DESC

/* 277. Issues whose status was changed by current user today - today's personal workflow actions */
status CHANGED BY currentUser() AFTER startOfDay() ORDER BY updated DESC

/* 278. Issues whose status was changed by current user this week - this week's workflow footprint */
status CHANGED BY currentUser() AFTER startOfWeek() ORDER BY updated DESC

/* 279. Issues escalated from Critical to Blocker this week - week's severity escalations */
priority CHANGED FROM Critical TO Blocker AFTER startOfWeek() AND resolution = Unresolved ORDER BY updated DESC

/* 280. Issues downgraded from Blocker before start of today - overnight priority reductions */
priority CHANGED FROM Blocker AFTER startOfDay("-1d") AND priority CHANGED BEFORE startOfDay() AND resolution = Unresolved ORDER BY updated DESC

/* 281. Issues transitioned from In Progress to In Review in the last 24 hours - review queue arrivals */
status CHANGED FROM "In Progress" TO "In Review" AFTER -1d ORDER BY updated DESC

/* 282. Issues moved to In Progress today by current user - today's personal sprint starts */
status CHANGED FROM Open TO "In Progress" AFTER startOfDay() BY currentUser() ORDER BY updated DESC

/* 283. Issues that had a fix version assigned for the first time this week - weekly scoping activity */
fixVersion CHANGED FROM EMPTY AFTER startOfWeek() AND resolution = Unresolved ORDER BY project ASC, updated DESC

/* 284. Issues assigned to someone for the first time today - today's first triage assignments */
assignee CHANGED FROM EMPTY AFTER startOfDay() ORDER BY priority ASC, updated DESC

/* 285. Issues resolved by current user today - today's personal closure count */
status CHANGED TO Resolved AFTER startOfDay() BY currentUser() ORDER BY updated DESC



/* worklogDate and worklogAuthor queries */

/* 286. Issues that had any time logged today - today's work log activity */
worklogDate >= startOfDay() ORDER BY updated DESC

/* 287. Issues with time logged this week - weekly work log scope */
worklogDate >= startOfWeek() AND resolution = Unresolved ORDER BY project ASC, updated DESC

/* 288. Issues with time logged this month - monthly work log breadth */
worklogDate >= startOfMonth() ORDER BY project ASC, updated DESC

/* 289. Issues with time logged in the last 7 days - rolling week work log */
worklogDate >= -7d ORDER BY updated DESC

/* 290. Issues where current user logged time today - today's personal work log */
worklogAuthor = currentUser() AND worklogDate >= startOfDay() ORDER BY updated DESC

/* 291. Issues where current user logged time this week - current user's weekly work log */
worklogAuthor = currentUser() AND worklogDate >= startOfWeek() ORDER BY project ASC, updated DESC

/* 292. Issues where current user logged time this month - current user's monthly work log */
worklogAuthor = currentUser() AND worklogDate >= startOfMonth() ORDER BY project ASC, updated DESC

/* 293. Issues in open sprints where time was logged today - today's sprint time entries */
worklogDate >= startOfDay() AND sprint in openSprints() ORDER BY project ASC, updated DESC

/* 294. Blocker or Critical issues that had time logged this week - high-priority active work */
worklogDate >= startOfWeek() AND priority in (Blocker, Critical) AND resolution = Unresolved ORDER BY priority ASC, updated DESC

/* 295. Issues current user resolved this year where they also logged time - personal annual fixed+logged */
worklogAuthor = currentUser() AND worklogDate >= startOfYear() AND resolution = Fixed ORDER BY resolved DESC



/* votes and watchers combined with time */

/* 296. Highly voted issues created this month - community-prioritised new issues */
votes > 5 AND created >= startOfMonth() AND resolution = Unresolved ORDER BY votes DESC, created DESC

/* 297. Issues with more than 10 votes due this month and still unresolved - popular upcoming deadline risk */
votes > 10 AND resolution = Unresolved AND duedate >= startOfMonth() AND duedate <= endOfMonth() ORDER BY votes DESC, duedate ASC

/* 298. Issues current user is watching that were updated today - watched issues with today's activity */
watcher = currentUser() AND updated >= startOfDay() ORDER BY updated DESC

/* 299. Issues current user is watching that are overdue and unresolved - watched overdue risk list */
watcher = currentUser() AND resolution = Unresolved AND duedate < now() ORDER BY duedate ASC, priority ASC

/* 300. Blocker issues with at least one vote created this week - new community-flagged blockers */
votes > 0 AND created >= startOfWeek() AND priority = Blocker AND resolution = Unresolved ORDER BY votes DESC, created DESC



/* linkedIssuesOf (direct, non-recursive) combined with time */

/* 301. Issues directly blocked by issue PROJ-100 */
issueFunction in linkedIssuesOf("issue = PROJ-100", "blocks") ORDER BY priority ASC

/* 302. Issues directly blocked by anything created this week */
issueFunction in linkedIssuesOf("created >= startOfWeek() AND resolution = Unresolved", "is blocked by") ORDER BY priority ASC, updated DESC

/* 303. Issues directly blocking items updated today */
issueFunction in linkedIssuesOf("updated >= startOfDay() AND resolution = Unresolved", "blocks") ORDER BY priority ASC

/* 304. Issues directly related to open sprint stories */
issueFunction in linkedIssuesOf("sprint in openSprints() AND issuetype = Story", "relates to") ORDER BY priority ASC, updated DESC

/* 305. Issues directly blocking work due today */
issueFunction in linkedIssuesOf("duedate <= endOfDay() AND resolution = Unresolved", "blocks") ORDER BY priority ASC, duedate ASC

/* 306. Issues directly blocking Blocker-priority unresolved issues */
issueFunction in linkedIssuesOf("priority = Blocker AND resolution = Unresolved", "is blocked by") ORDER BY priority ASC, updated DESC

/* 307. Issues directly relating to In Progress stories in open sprints */
issueFunction in linkedIssuesOf("status = 'In Progress' AND sprint in openSprints()", "relates to") ORDER BY updated DESC

/* 308. Issues directly depending on Critical issues in unreleased versions */
issueFunction in linkedIssuesOf("fixVersion in unreleasedVersions() AND priority = Critical AND resolution = Unresolved", "depends on") ORDER BY duedate ASC

/* 309. Issues directly caused by something resolved today */
issueFunction in linkedIssuesOf("resolved >= startOfDay() AND resolution IS NOT EMPTY", "is caused by") ORDER BY priority ASC

/* 310. Issues directly duplicated by bugs created this month */
issueFunction in linkedIssuesOf("created >= startOfMonth() AND issuetype = Bug", "duplicates") ORDER BY priority ASC, created DESC



/* Complex multi-condition queries combining issueFunction, time, sprint, version, and worklog */

/* 311. Epics containing Blocker stories in open sprints, themselves updated this week - active risky epics */
issueFunction in epicsOf("sprint in openSprints() AND priority = Blocker AND resolution = Unresolved") AND updated >= startOfWeek() ORDER BY updated DESC

/* 312. Issues inside epics targeting unreleased versions, due this month, still unresolved */
issueFunction in issuesInEpics("fixVersion in unreleasedVersions() AND resolution = Unresolved") AND duedate <= endOfMonth() AND resolution = Unresolved ORDER BY duedate ASC, priority ASC

/* 313. Parent issues of subtasks created in the last 24 hours, currently in open sprints */
issueFunction in parentsOf("issuetype = Sub-task AND created >= -1d") AND sprint in openSprints() AND resolution = Unresolved ORDER BY priority ASC

/* 314. All issues recursively blocked by In Progress stories in open sprints, created this month */
issueFunction in linkedIssuesOfRecursive("sprint in openSprints() AND status = 'In Progress'", "blocks") AND created >= startOfMonth() AND resolution = Unresolved ORDER BY priority ASC

/* 315. Subtasks of earliest-unreleased-version issues updated today, still unresolved */
issueFunction in subtasksOf("fixVersion = earliestUnreleasedVersion() AND updated >= startOfDay()") AND resolution = Unresolved ORDER BY priority ASC

/* 316. Issues in open sprints with subtasks where work was logged this week */
worklogDate >= startOfWeek() AND issueFunction in hasSubtasks() AND sprint in openSprints() AND resolution = Unresolved ORDER BY priority ASC, updated DESC

/* 317. Issues current user commented on this week, in open sprints, still unresolved */
issueFunction in commented("by currentUser() after startOfWeek()") AND sprint in openSprints() AND resolution = Unresolved ORDER BY priority ASC, updated DESC

/* 318. Issues in epics with overdue due dates, targeting unreleased fix versions */
issueFunction in epicsOf("duedate < now() AND resolution = Unresolved") AND fixVersion in unreleasedVersions() AND resolution = Unresolved ORDER BY duedate ASC

/* 319. Open sprint issues escalated to Blocker this week that are recursively blocked by release-blockers */
sprint in openSprints() AND priority CHANGED TO Blocker AFTER startOfWeek() AND issueFunction in linkedIssuesOfRecursive("labels = 'release-blocker'", "is blocked by") ORDER BY updated DESC

/* 320. Parent issues of subtasks where time was logged today, in open sprints */
issueFunction in parentsOf("worklogDate >= startOfDay() AND issuetype = Sub-task") AND sprint in openSprints() AND resolution = Unresolved ORDER BY priority ASC, updated DESC

/* 321. Issues in unreleased versions directly blocking release-blocker issues, due this month */
fixVersion in unreleasedVersions() AND issueFunction in linkedIssuesOf("labels = 'release-blocker'", "blocks") AND duedate <= endOfMonth() AND resolution = Unresolved ORDER BY duedate ASC

/* 322. Issues current user is watching, recursively caused by incidents created in the last 24 hours */
watcher = currentUser() AND issueFunction in linkedIssuesOfRecursive("issuetype = Incident AND created >= -1d", "is caused by") ORDER BY priority ASC, updated DESC

/* 323. Issues with logged work this week, targeting unreleased versions, linked to any blocking issue */
worklogDate >= startOfWeek() AND fixVersion in unreleasedVersions() AND issueFunction in hasLinks("blocks") AND resolution = Unresolved ORDER BY priority ASC, updated DESC

/* 324. Issues in open sprints transitioned back to In Progress from In Review today, with subtasks */
sprint in openSprints() AND status CHANGED FROM "In Review" TO "In Progress" AFTER startOfDay() AND issueFunction in hasSubtasks() ORDER BY updated DESC

/* 325. Recursive depends-on and blocked-by chain from in-progress epics in open sprints, created this month */
issueFunction in linkedIssuesOfRecursive("sprint in openSprints() AND issuetype = Epic AND status = 'In Progress'", "depends on", "is blocked by") AND created >= startOfMonth() AND resolution = Unresolved ORDER BY priority ASC, duedate ASC

/* 326. Complex query using JQL fields and date functions - Show recently created high‑priority bugs in HADOOP that were updated in the last 24 hours */
project = HADOOP AND issuetype = Bug AND priority IN (Blocker, Critical, High) AND created >= startOfDay(-7d) AND updated >= -24h ORDER BY priority DESC, updated DESC

/* 327. Complex query using JQL operators and boolean logic - Show issues in HADOOP or SPARK that are open or In Progress and not low‑priority */
(project = HADOOP OR project = SPARK) AND status IN ("To Do", "In Progress", "Open") AND priority NOT IN (Low) AND text ~ "performance" ORDER BY project ASC, priority DESC

/* 328. Complex query using JQL keywords and functions - Show issues created by the current user in the current month, sorted by creation */
reporter = currentUser() AND created >= startOfMonth() AND created <= endOfMonth() AND status NOT IN (Done, Closed) ORDER BY created DESC

/* 329. Complex query using JQL developer‑status / teams fields - Show issues assigned to a specific developer or team that are unresolved and have been in progress for more than 7 days */
assignee = devTeam OR "Developer Team" = devTeam AND status = "In Progress" AND status CHANGED TO "In Progress" BEFORE startOfDay(-7d) AND resolution = Unresolved ORDER BY created ASC

/* 330. Complex query using JQL for custom fields and roadmaps - Show issues in Advanced Roadmaps‑style custom fields where “Release Blocker” is true and the issue is high‑priority */
"Release Blocker" = true AND priority IN (Blocker, Critical, High) AND project = "ROADMAP-PROJ" AND labels = "migration" AND issueFunction IN epicsOf("resolution = Unresolved") ORDER BY "Release Blocker" DESC, priority DESC

/* 331. Issues whose status was "In Progress" at the start of today - Show issues that were in "In Progress" at the beginning of today */
status WAS "In Progress" ON startOfDay() ORDER BY project ASC, created DESC

/* 332. Issues moved to "Done" in the last 3 days - Show issues resolved in the last 3 days ordered by project and priority */
status CHANGED TO "Done" AFTER startOfDay(-3d) ORDER BY project ASC, priority DESC

/* 333. Issues with comments added in the last hour - Show recently commented issues across projects */
comment ~ "comment" AND commentDate >= -1h ORDER BY updated DESC

/* 334. High‑priority bugs without labels - Show high‑severity bugs that have no labels */
issuetype = Bug AND priority IN (Blocker, Critical) AND labels IS EMPTY ORDER BY project ASC

/* 335. Issues not in any sprint - Show issues not yet scheduled into a sprint */
sprint IS EMPTY ORDER BY created DESC

/* 336. Issues in the current sprint for a specific project - Show all issues in the current sprint of HADOOP */
project = HADOOP AND sprint = "Current Sprint" ORDER BY status ASC

/* 337. Issues linked to any bug - Show issues that are linked to at least one bug */
issueFunction IN linkedIssuesOf("issuetype = Bug", "", "is blocked by") ORDER BY priority DESC

/* 338. Bugs that block other issues - Show bugs that block at least one other issue */
issuetype = Bug AND issueFunction IN linkedIssuesOf("","", "blocks") ORDER BY project ASC

/* 339. Issues whose status was "In Progress" in the last 7 days - Show issues that were in "In Progress" at some point in the last week */
status WAS "In Progress" AFTER startOfDay(-7d) ORDER BY created DESC

/* 340. Issues whose status changed to "In Progress" more than 2 times - Show issues that switched to "In Progress" frequently */
status CHANGED TO "In Progress" AFTER -30d AND status CHANGED COUNT > 2 ORDER BY priority DESC

/* 341. Issues updated in the last 15 minutes - Show very recently touched issues */
updated >= -15m ORDER BY updated DESC

/* 342. Issues created in the last 48 hours - Show issues created in the last 2 days ordered by project */
created >= -48h ORDER BY project ASC

/* 343. Issues with empty description - Show issues missing detailed descriptions */
description IS EMPTY ORDER BY updated DESC

/* 344. Issues with attachments - Show issues that have at least one attachment */
attachment IS NOT EMPTY ORDER BY project ASC

/* 345. Issues with no attachments - Show issues without any attachments */
attachment IS EMPTY ORDER BY created DESC

/* 346. Issues with more than 5 comments - Show issues with extensive discussion */
comment SIZE > 5 ORDER BY commentSize DESC

/* 347. Issues with no comments - Show issues that have never been commented on */
comment IS EMPTY ORDER BY updated DESC

/* 348. Issues in specific components - Show issues related to both HDFS and YARN */
project = HADOOP AND component IN (HDFS, YARN) AND component SIZE > 1 ORDER BY project ASC

/* 349. Issues assigned to a specific user who created them - Show issues owned by the same user who reported them */
reporter = assignee ORDER BY created DESC

/* 350. Issues with custom field text match - Show issues where "Impact" contains "high" */
"Impact" ~ "high" ORDER BY project ASC

/* 351. Issues with multiple labels - Show issues tagged with more than one label */
labels IN (performance, regression) AND labels SIZE > 1 ORDER BY created DESC

/* 352. Issues that are part of an epic - Show all issues linked to an epic */
"Epic Link" IS NOT EMPTY ORDER BY project ASC

/* 353. Epics that still have unresolved issues - Show epics that are not fully closed */
issuetype = Epic AND issueFunction IN epicsOf("resolution = Unresolved") ORDER BY project ASC

/* 354. Stories under unresolved epics - Show stories whose epic is still open */
issuetype = Story AND issueFunction IN epicsOf("resolution = Unresolved") ORDER BY project ASC

/* 355. Subtasks whose parent is unresolved - Show subtasks whose parent is still open */
issuetype = Sub-task AND parent IN issueFunction subtasksOf("status != Done") ORDER BY created DESC

/* 356. Issues blocked by bugs - Show issues that are blocked by a bug */
issueFunction IN linkedIssuesOf("issuetype = Bug", "", "is blocked by") ORDER BY priority DESC

/* 357. Bugs that block issues - Show bugs that block other issues */
issuetype = Bug AND issueFunction IN linkedIssuesOf("","", "blocks") ORDER BY project ASC

/* 358. Issues blocked by a specific issue - Show all issues blocked by HADOOP-123 */
issueLink = HADOOP-123 AND issueLinkType = "is blocked by" ORDER BY project ASC

/* 359. Issues blocked by high‑priority issues - Show issues blocked by Blocker or Critical issues */
issueFunction IN linkedIssuesOf("priority IN (Blocker, Critical)", "", "is blocked by") ORDER BY project ASC

/* 360. Issues blocking features - Show issues that block at least one feature */
issuetype = Bug AND issueFunction IN linkedIssuesOf("issuetype = Feature", "", "blocks") ORDER BY project ASC

/* 361. Features blocked by bugs - Show features that are blocked by bugs */
issuetype = Feature AND issueFunction IN linkedIssuesOf("issuetype = Bug", "", "is blocked by") ORDER BY project ASC

/* 362. Issues updated more than 7 days ago and still open - Show stale open issues created recently */
updated < -7d AND resolution = Unresolved ORDER BY updated ASC

/* 363. Issues due in the next 3 days - Show issues due within the next 72 hours */
duedate >= startOfDay() AND duedate <= endOfDay(3d) ORDER BY duedate ASC

/* 364. Issues overdue (due before today) - Show overdue issues ordered by project */
duedate < startOfDay() AND resolution = Unresolved ORDER BY project ASC

/* 365. Issues in future sprints - Show issues planned for future sprints */
sprint IN futureSprints() ORDER BY project ASC

/* 366. Issues with fixVersion not set - Show issues not yet assigned to a release */
fixVersion IS EMPTY ORDER BY created DESC

/* 367. Issues targeted for a specific release - Show issues assigned to Hadoop 4.0.0 */
project = HADOOP AND fixVersion = "Hadoop 4.0.0" ORDER BY project ASC

/* 368. Unresolved bugs with no fixVersion - Show open bugs not yet scheduled for a release */
issuetype = Bug AND resolution = Unresolved AND fixVersion IS EMPTY ORDER BY project ASC

/* 369. Issues with comments from a specific user - Show issues with comments by Anna */
comment ~ "Anna" ORDER BY updated DESC

/* 370. Issues with attachments added in the last 24 hours - Show issues with recent attachments */
attachment CHANGED AFTER -24h ORDER BY updated DESC

/* 371. Issues with no labels and no attachments - Show minimal issues with no metadata */
labels IS EMPTY AND attachment IS EMPTY ORDER BY created DESC

/* 372. Issues updated in the current quarter - Show issues updated in the current quarter */
updated >= startOfQuarter() AND updated <= endOfQuarter() ORDER BY project ASC

/* 373. Issues created in the current year - Show issues created this year */
created >= startOfYear() ORDER BY project ASC

/* 374. Issues whose priority changed in the last 7 days - Show issues with recent priority updates */
priority WAS CHANGED AFTER -7d ORDER BY project ASC

/* 375. Issues whose status changed multiple times in the last 14 days - Show issues with frequent status changes */
status CHANGED AFTER -14d AND status CHANGED COUNT > 3 ORDER BY statusChangedCount DESC

/* 376. Issues with text in summary or description - Show issues mentioning "security" anywhere */
text ~ "security" ORDER BY project ASC

/* 377. Issues with specific custom field value and label - Show issues with "Deployment Env" = "Production" and label "performance" */
"Deployment Env" = Production AND labels = performance ORDER BY project ASC

/* 378. Issues with any custom field matching text - Show issues where "Notes" contains "urgent" */
"Notes" ~ "urgent" ORDER BY project ASC

/* 379. Issues whose assignee changed in the last 7 days - Show issues recently reassigned */
assignee WAS CHANGED AFTER -7d ORDER BY project ASC

/* 380. Issues whose status was "In Progress" more than 7 days ago - Show issues that were in "In Progress" over a week ago */
status WAS "In Progress" BEFORE startOfDay(-7d) ORDER BY project ASC

/* 381. Issues whose status changed to "In Progress" and is still open - Show issues that entered "In Progress" recently and are still open */
status CHANGED TO "In Progress" AFTER -7d AND status = "In Progress" ORDER BY project ASC

/* 382. Issues with label "urgent" and high priority - Show urgent high‑priority issues across projects */
labels = urgent AND priority IN (Blocker, Critical, High) ORDER BY project ASC

/* 383. Issues with more than 3 labels - Show heavily tagged issues */
labels SIZE > 3 ORDER BY project ASC

/* 384. Issues with comments added in the last 10 minutes - Show very recently commented issues */
commentDate >= -10m ORDER BY updated DESC

/* 385. Issues whose status never changed from "To Do" - Show issues stuck in "To Do" */
status WAS "To Do" AND status = "To Do" AND status CHANGED COUNT = 1 ORDER BY created ASC

/* 386. Issues with fixVersion in a list and high priority - Show issues targeted for specific releases and high priority */
project = HADOOP AND fixVersion IN ("Hadoop 4.0.0", "Hadoop 4.1.0") AND priority IN (Blocker, Critical) ORDER BY project ASC

/* 387. Issues with custom field "Risk Level" = "High" - Show high‑risk issues */
"Risk Level" = High ORDER BY project ASC

/* 388. Issues with custom field "Risk Level" updated in the last week - Show issues with recent risk‑level changes */
"Risk Level" WAS CHANGED AFTER -7d ORDER BY project ASC

/* 389. Issues with attachments updated in the last 7 days - Show issues with recent attachment changes */
attachment CHANGED AFTER -7d ORDER BY project ASC

/* 390. Issues with comments from the current user - Show issues commented on by the logged‑in user */
comment ~ currentUser() ORDER BY updated DESC

/* 391. Issues with no fixVersion and high priority - Show high‑priority issues not yet assigned to a release */
priority IN (Blocker, Critical) AND fixVersion IS EMPTY ORDER BY project ASC

/* 392. Issues with label "regression" and bug type - Show regressions of type bug */
issuetype = Bug AND labels = regression ORDER BY project ASC

/* 393. Issues with label "regression" and any epic link - Show regression‑labelled issues that are part of an epic */
labels = regression AND "Epic Link" IS NOT EMPTY ORDER BY project ASC

/* 394. Issues whose status changed to "Done" and then back - Show issues that bounced out of "Done" */
status WAS "Done" AND status != "Done" ORDER BY project ASC

/* 395. Issues with specific component and label - Show issues in HDFS with label "regression" */
component = HDFS AND labels = regression ORDER BY project ASC

/* 396. Issues with specific component and status - Show issues in YARN that are "In Progress" */
component = YARN AND status = "In Progress" ORDER BY project ASC

/* 397. Issues with specific component and created in last 7 days - Show recently created HDFS issues */
component = HDFS AND created >= -7d ORDER BY project ASC

/* 398. Issues with specific component and updated in last 24 hours - Show recently updated HDFS issues */
component = HDFS AND updated >= -24h ORDER BY project ASC

/* 399. Issues with specific label and created in last 3 days - Show recently created "performance"‑labelled issues */
labels = performance AND created >= -3d ORDER BY project ASC

/* 400. Issues with specific label and updated in last 24 hours - Show recently updated "performance"‑labelled issues */
labels = performance AND updated >= -24h ORDER BY project ASC

/* 401. Show up to 10 recently updated Major/Minor issues in ZOOKEEPER */
project = ZOOKEEPER AND priority IN (Major, Minor) ORDER BY updated DESC

/* 402. Show up to 10 recently created issues in ZOOKEEPER */
project = ZOOKEEPER ORDER BY created DESC

/* 403. Show up to 10 open High/Critical issues in ZOOKEEPER */
project = ZOOKEEPER AND priority IN (Critical, High) AND status != Done ORDER BY priority DESC

/* 404. Show up to 10 issues assigned to the current user in ZOOKEEPER */
project = ZOOKEEPER AND assignee = currentUser() ORDER BY updated DESC

/* 405. Show up to 10 unresolved bugs in ZOOKEEPER */
project = ZOOKEEPER AND issuetype = Bug AND resolution = Unresolved ORDER BY created DESC

/* 406. Show up to 10 issues created in the last 7 days in ZOOKEEPER */
project = ZOOKEEPER AND created >= -7d ORDER BY created DESC

/* 407. Show up to 10 issues updated in the last 24 hours in ZOOKEEPER */
project = ZOOKEEPER AND updated >= -24h ORDER BY updated DESC

/* 408. Show up to 10 issues with subtasks in ZOOKEEPER */
project = ZOOKEEPER AND issueFunction IN hasSubtasks() ORDER BY priority DESC

/* 409. Show up to 10 issues with no subtasks in ZOOKEEPER */
project = ZOOKEEPER AND issueFunction NOT IN hasSubtasks() ORDER BY created DESC

/* 410. Show up to 10 issues linked to unresolved bugs in ZOOKEEPER */
project = ZOOKEEPER AND issueFunction IN linkedIssuesOf("issuetype = Bug AND resolution = Unresolved", "", "is blocked by") ORDER BY updated DESC

/* 411. Open issues assigned to me, highest priority first */
assignee = currentUser() AND resolution IS EMPTY ORDER BY priority DESC, updated DESC

/* 412. All bugs in project KAFKA created in the last 30 days */
project = KAFKA AND issuetype = Bug AND created >= -30d ORDER BY created DESC

/* 413. Stories in ZOOKEEPER that are In Progress or In Review */
project = ZOOKEEPER AND issuetype = Story AND status IN ("In Progress", "In Review") ORDER BY updated DESC

/* 414. Major or Critical issues resolved this month */
priority IN (Major, Critical) AND resolutiondate >= startOfMonth() AND resolutiondate <= endOfMonth() ORDER BY resolutiondate DESC

/* 415. Issues without an assignee that are not Done */
assignee IS EMPTY AND statusCategory != Done ORDER BY created ASC

/* 416. List all supported issue types in project KAFKA (distinct in results view) */
project = KAFKA ORDER BY issuetype ASC

/* 417. Combined issues from ZOOKEEPER and HIVE, newest first */
project IN (ZOOKEEPER, HIVE) ORDER BY created DESC

/* 418. Open issues from ZOOKEEPER and HIVE, sorted by project then key */
project IN (ZOOKEEPER, HIVE) AND resolution IS EMPTY ORDER BY project, key

/* 419. Stories blocked by open bugs in core platform projects */ issueFunction in linkedIssuesOf("project in (HIVE, KAFKA, ZOOKEEPER) AND issuetype = Bug AND status in (\"In Progress\", \"Open\")", "blocks") AND issuetype in (Story, Task) ORDER BY priority DESC, updated DESC

/* 420. Dependency chain up to 3 levels deep for a given release epic */ issueFunction in linkedIssuesOfRecursiveLimited("issue = HIVE-123 AND issuetype = Epic", 3, "is blocked by") ORDER BY project, issuetype, key

/* 421. Cross-project dependencies into HIVE from KAFKA and ZOOKEEPER */ project = HIVE AND issueFunction in linkedIssuesOf("project in (KAFKA, ZOOKEEPER) AND statusCategory != Done", "is blocked by") ORDER BY statusCategory, updated DESC

/* 422. Open issues that block at least one unresolved issue in any of the three projects */ issueFunction in linkedIssuesOf("project in (HIVE, KAFKA, ZOOKEEPER) AND resolution IS EMPTY", "is blocked by") AND resolution IS EMPTY ORDER BY priority DESC, created ASC

/* 423. All issues within 2-link distance of a critical production incident */ issueFunction in linkedIssuesOfRecursiveLimited("issue = KAFKA-999 AND priority = Highest", 2) ORDER BY project, priority DESC, updated DESC

/* 424. Epics/Features that have at least one linked Story/Task/Sub-task */
issuetype in (Epic, Feature) AND issueFunction in linkedIssuesOf("issuetype in (Story, Task, Sub-task)", "has Epic") ORDER BY updated DESC

/* 425. Stories/Tasks that are not linked to any Epic/Feature (broken hierarchy) */ 
issuetype in (Story, Task) AND issueFunction not in linkedIssuesOf("issuetype in (Epic, Feature)", "has Epic") ORDER BY created DESC

/* 426. Sub-tasks whose parent Story/Task belongs to a specific Epic/Feature */ 
issuetype = Sub-task AND issueFunction in linkedIssuesOfRecursiveLimited("issue = HIVE-123 AND issuetype in (Epic, Feature)", 2) ORDER BY priority DESC, updated DESC

/* 427. All Stories/Tasks/Sub-tasks under Epics/Features in HIVE, KAFKA, ZOOKEEPER */ 
issuetype in (Story, Task, Sub-task) AND issueFunction in linkedIssuesOfRecursiveLimited("project in (HIVE, KAFKA, ZOOKEEPER) AND issuetype in (Epic, Feature)", 2) ORDER BY project, issuetype, key

/* 428. Epics/Features that have at least one blocked Story/Task/Sub-task */ 
issuetype in (Epic, Feature) AND issueFunction in linkedIssuesOfRecursiveLimited("issuetype in (Story, Task, Sub-task) AND status in (\"Blocked\", \"On Hold\")", 2) ORDER BY priority DESC, updated DESC

/* 429. Fetch a single issue by its key */
issue = KAFKA-20404

/* 430. Show details for a specific ticket */
issue = ZOOKEEPER-100

/* 431. Look up a single Jira issue by key to view its fields */
issue = HIVE-5000

/* 432. List all the issues in the sprints */
Sprint is not EMPTY

/* 433. List all the issues in sprint Usergrid 36 */
Sprint = "Usergrid 36"

/* 434. List all the issues in sprint id 224 */
Sprint = 224

/* 435. List all the issues in sprint id 224,225,226 */
Sprint in (224,225,226)

/* 436. List linked issues of type Bug, Story, Task, Sub-task in project KAFKA, ZOOKEEPER, or FLINK */
(project in (KAFKA, ZOOKEEPER, FLINK)) AND issueFunction in linkedIssuesOf("issuetype IN (Bug, Story, Task, 'Sub-task', Improvement)") ORDER BY created DESC


/* 437. Open Bugs linked to issues in KAFKA/ZOOKEEPER/FLINK projects */
(project in (KAFKA, ZOOKEEPER, FLINK)) AND issueFunction in linkedIssuesOf("project in (KAFKA, ZOOKEEPER, FLINK)") AND issuetype = Bug AND resolution is EMPTY ORDER BY priority DESC


/* 438. Recent Tasks linked from KAFKA project only */
project = KAFKA AND issueFunction in linkedIssuesOf("project = KAFKA") AND issuetype = Task ORDER BY created DESC


/* 439. Improvements blocking KAFKA/ZOOKEEPER/FLINK issues */
(project in (KAFKA, ZOOKEEPER, FLINK)) AND issueFunction in linkedIssuesOf("project in (KAFKA, ZOOKEEPER, FLINK)", blocks) AND issuetype = Improvement ORDER BY updated DESC


/* 440. High priority issues linked to FLINK Bugs */
project = FLINK AND issueFunction in linkedIssuesOf("project = FLINK AND issuetype = Bug") AND priority in (Blocker, Critical) ORDER BY priority ASC, created DESC


/* 441. All linked issues from ZOOKEEPER project regardless of type */
project in (KAFKA, ZOOKEEPER, FLINK) AND issueFunction in linkedIssuesOf("project = ZOOKEEPER") ORDER BY created DESC 


/* 442. Bugs fixed recently that link to open issues */
(project in (KAFKA, ZOOKEEPER, FLINK)) AND issuetype = Bug AND resolution = Fixed  AND issueFunction in linkedIssuesOf("resolution is EMPTY") AND updated >= -30d ORDER BY updated DESC


/* 444. Tasks assigned to current user with project links */
issuetype = Task AND project in (KAFKA, ZOOKEEPER, FLINK) AND issueFunction in linkedIssuesOf("project in (KAFKA, ZOOKEEPER, FLINK)") ORDER BY status, created DESC


/* 445. Duplicate issues linked across KAFKA/FLINK projects */
(project in (KAFKA, FLINK)) AND issueFunction in linkedIssuesOf("project in (KAFKA, FLINK)", duplicates) ORDER BY created DESC


/* 446. List issues from projects KAFKA, HIVE, or ZOOKEEPER that have a direct issue link of any issue type*/
project in (KAFKA, HIVE, ZOOKEEPER) AND issueLinkType in (  "blocks",  "is blocked by",  "relates to",  "duplicates",  "is duplicated by",  "is cloned by")

/* 447. List issues from projects KAFKA, HIVE, or ZOOKEEPER that have a at least 1 issue type link or linked issues of any type */
project in (KAFKA, HIVE, ZOOKEEPER) AND issueLinkType in (  "blocks",  "is blocked by",  "relates to",  "duplicates",  "is duplicated by",  "is cloned by")

/* 448. In project KAFKA, find Epics that have linked Stories, and Stories that in turn have linked Tasks or Sub-tasks. Use this as the base set; build Epic → Story → Task/Sub-task link chains in post-processing. */
project = KAFKA AND issueLinkType in (  "blocks",  "is blocked by",  "relates to",  "duplicates",  "is duplicated by",  "is cloned by") and issuetype in (Story, Task, Sub-task, Epic)

/* 449. List all closed issues in projects KAFKA and HIVE from Q1 2026 (January–March), sorted by resolution date */
project in (KAFKA, HIVE)  AND status = Closed  AND resolutiondate >= "2026-01-01"  AND resolutiondate <= "2026-03-31" ORDER BY resolutiondate DESC

/* 450. Bugs in project FLINK fixed in 2025, ordered by resolution date */
project = FLINK AND issuetype = Bug AND resolution = Fixed AND resolutiondate >= "2025-01-01" AND resolutiondate <= "2025-12-31" ORDER BY resolutiondate DESC

/* 451. Issues from project KAFKA created in Q1 2025 (Jan–Mar), newest first */
project = KAFKA AND created >= "2025-01-01" AND created <= "2025-03-31" ORDER BY created DESC

/* 452. Open HIVE issues from 2024 (all four quarters), by project then creation date */
project = HIVE AND resolution IS EMPTY AND created >= "2024-01-01" AND created <= "2024-12-31" ORDER BY project, created ASC

/* 453. List all issues resolved in projects KAFKA and HIVE in 2026 quarter 1 */
project in (KAFKA, HIVE)  AND status = Closed  AND resolutiondate >= "2026-01-01"  AND resolutiondate <= "2026-03-31" ORDER BY resolutiondate DESC


/* 454. List all issues resolved or closed or solved or done or fixed in projects KAFKA and HIVE in 2026 quarter 1 */
project in (KAFKA, HIVE)  and resolution in (Resolved, Fixed) OR resolutiondate >= "2026-01-01"  AND resolutiondate <= "2026-03-31" ORDER BY resolutiondate DESC

/* 455. show the assignee who has solved maximum number of bugs in project kafka */
project = KAFKA  AND issuetype = Bug  AND status in (Closed, Resolved) ORDER BY assignee ASC

/* 456. show the assignee who has solved maximum number of issues in project kafka */
project = KAFKA AND status in (Closed,Done,Resolved) ORDER BY assignee ASC

/* 456. Show all unresolved blockers in KAFKA with no assignee */
project = KAFKA AND priority = Blocker AND status not in (Closed, Resolved) AND assignee is EMPTY ORDER BY created ASC

/* 457. Show all bugs reported by me in KAFKA or HIVE that are still open */
project in (KAFKA, HIVE) AND issuetype = Bug AND reporter = currentUser() AND status not in (Closed, Resolved) ORDER BY created DESC

/* 458. Show all issues closed in KAFKA last 90 days that were originally blocker priority */
project = KAFKA AND priority = Blocker AND status in (Closed, Resolved) AND resolutiondate >= -90d ORDER BY resolutiondate DESC

/* 459. Show all open issues in KAFKA assigned to me that have not been updated in over 14 days */
project = KAFKA AND assignee = currentUser() AND status not in (Closed, Resolved) AND updated <= -14d ORDER BY updated ASC

/* 460. Show all tasks and sub-tasks in FLINK that are linked to an unresolved bug in KAFKA */
issuetype in (Task, Sub-task) AND project = FLINK AND issueFunction in linkedIssuesOf("project = KAFKA AND issuetype = Bug AND status not in (Closed, Resolved)") ORDER BY created DESC

/* 461. Show all issues in ZOOKEEPER due this week that are not yet resolved */
project = ZOOKEEPER AND due >= startOfWeek() AND due <= endOfWeek() AND status not in (Closed, Resolved) ORDER BY due ASC

/* 462. Show all improvements and new features added to KAFKA in the last 6 months */
project = KAFKA AND issuetype in (Improvement, "New Feature") AND created >= -180d AND status in (Closed, Resolved) ORDER BY resolutiondate DESC

/* 463. Show all issues in HIVE where fix version is 4.0.0 and are still unresolved */
project = HIVE AND fixVersion = "4.0.0" AND status not in (Closed, Resolved) ORDER BY priority DESC, created ASC

/* 464. Show all critical bugs across KAFKA, HIVE, FLINK reopened in the last 30 days */
project in (KAFKA, HIVE, FLINK) AND issuetype = Bug AND priority = Critical AND status = Reopened AND updated >= -30d ORDER BY updated DESC

/* 465. Show all issues assigned to me across all Apache projects that are overdue and unresolved */
assignee = currentUser() AND due < now() AND status not in (Closed, Resolved) ORDER BY due ASC, priority DESC

/* 466. Show all tasks and sub-tasks in HADOOP that are linked to an unresolved bug in HIVE */
issuetype in (Task, Sub-task) AND project = HADOOP AND issueFunction in linkedIssuesOf("project = HIVE AND issuetype = Bug AND status not in (Closed, Resolved)") ORDER BY created DESC

/* 466. Show all unresolved issues in the current active sprint for KAFKA */
project = KAFKA AND sprint in openSprints() AND status not in (Closed, Resolved) ORDER BY priority DESC, updated ASC

/* 467. Show all issues that were not completed and carried over from the last sprint to current sprint in KAFKA */
project = KAFKA AND sprint in closedSprints() AND sprint in openSprints() AND status not in (Closed, Resolved) ORDER BY priority DESC

/* 468. Show all bugs added to the current sprint after it started (mid-sprint additions) in KAFKA and HIVE */
project in (KAFKA, HIVE) AND issuetype = Bug AND sprint in openSprints() AND created >= sprintStart() ORDER BY created ASC

/* 469. Show all issues completed in the last 3 closed sprints across KAFKA, HIVE, FLINK */
project in (KAFKA, HIVE, FLINK) AND sprint in closedSprints() AND status in (Closed, Resolved) AND resolutiondate >= -42d ORDER BY resolutiondate DESC

/* 470. Show all critical and blocker issues assigned to current user due before end of current sprint in KAFKA */
project = KAFKA AND assignee = currentUser() AND priority in (Critical, Blocker) AND sprint in openSprints() AND due <= endOfWeek() AND status not in (Closed, Resolved) ORDER BY priority DESC, due ASC

/* 471. Show all issues that breached SLA by remaining unresolved for more than 30 days after their due date in KAFKA */
project = KAFKA AND due <= -30d AND status not in (Closed, Resolved) ORDER BY due ASC, priority DESC

/* 472. Show month-over-month regression — bugs reopened in KAFKA and HIVE that were previously resolved in the last 6 months */
project in (KAFKA, HIVE) AND issuetype = Bug AND status = Reopened AND resolutiondate >= -180d AND resolutiondate <= now() ORDER BY resolutiondate ASC, priority DESC

/* 473. Show all blocker and critical issues in KAFKA that took more than 60 days to resolve in 2025 — identifying long-running bottlenecks */
project = KAFKA AND priority in (Blocker, Critical) AND status in (Closed, Resolved) AND created <= "2025-01-01" AND resolutiondate >= "2025-01-01" AND resolutiondate <= "2025-12-31" ORDER BY resolutiondate DESC

/* 474. Show quarterly delivery health — all issues committed to fix version 3.9.0 in KAFKA that missed the release and remain unresolved */
project = KAFKA AND fixVersion = "3.9.0" AND status not in (Closed, Resolved) AND due < now() ORDER BY priority DESC, due ASC

/* 475. Show executive risk summary — all unresolved blockers and criticals across KAFKA, HIVE, FLINK, ZOOKEEPER older than 45 days with no update in the last 7 days indicating stalled delivery */
project in (KAFKA, HIVE, FLINK, ZOOKEEPER) AND priority in (Blocker, Critical) AND status not in (Closed, Resolved) AND created <= -45d AND updated <= -7d ORDER BY created ASC, priority DESC

/* 476. Show all bugs that were resolved by developers but not yet verified by QA, 
sitting unverified for more than 5 days across KAFKA and HIVE — indicating QA pipeline backlog */
project in (KAFKA, HIVE) AND issuetype = Bug AND status = Resolved AND updated <= -5d 
AND resolution = Fixed AND assignee != currentUser() 
ORDER BY updated ASC, priority DESC

/* 477. Show all issues targeting the next two upcoming fix versions in KAFKA 
that are still unresolved within 14 days of their due date — flagging release readiness risk */
project = KAFKA AND fixVersion in unreleasedVersions() AND status not in (Closed, Resolved) 
AND due <= 14d AND due >= now() 
ORDER BY fixVersion ASC, priority DESC, due ASC

/* 478. Show all issues in the active sprint across KAFKA and FLINK 
that have not been updated in more than 3 days — identifying blocked or idle work items */
project in (KAFKA, FLINK) AND sprint in openSprints() AND updated <= -3d 
AND status not in (Closed, Resolved) AND assignee is not EMPTY 
ORDER BY updated ASC, priority DESC

/* 479. Show all issues assigned to me across KAFKA, HIVE, FLINK 
that are due in the next 7 days, are in progress, and have linked blocking dependencies 
still unresolved — helping prioritize work with upstream blockers */
project in (KAFKA, HIVE, FLINK) AND assignee = currentUser() AND status = "In Progress" 
AND due >= now() AND due <= 7d 
AND issueFunction in linkedIssuesOf("status not in (Closed, Resolved) AND issueFunction in linkedIssuesOf(\"type = Blocker\")") 
ORDER BY due ASC, priority DESC

/* 480. Show all unresolved issues across the entire Apache portfolio 
that have been open for more than 180 days, are blocker or critical priority, 
have changed assignee more than once indicating ownership churn, 
and belong to active fix versions — representing the highest strategic delivery risk */
project in (KAFKA, HIVE, FLINK, ZOOKEEPER) AND priority in (Blocker, Critical) 
AND status not in (Closed, Resolved) AND created <= -180d 
AND fixVersion in unreleasedVersions() AND assignee is not EMPTY 
ORDER BY created ASC, priority DESC, updated ASC


/* 481. Show all security-related bugs across KAFKA, HIVE, ZOOKEEPER 
that have been unresolved for more than 72 hours since creation, are blocker or critical, 
have no fix version committed, and were not updated in the last 24 hours — 
indicating unacknowledged security vulnerabilities with no remediation plan */
project in (KAFKA, HIVE, ZOOKEEPER) 
AND issuetype = Bug 
AND labels in (security, vulnerability, CVE) 
AND priority in (Blocker, Critical) 
AND created <= -3d 
AND updated <= -1d 
AND fixVersion is EMPTY 
AND status not in (Closed, Resolved) 
ORDER BY created ASC, priority DESC

/* 482. Show all epics across KAFKA, HIVE, FLINK 
that were due in the last 60 days but are still unresolved, 
have at least one child issue still open, 
belong to an unreleased fix version, 
and have not received any update in the last 10 days — 
representing stalled program-level deliverables that missed their window */
project in (KAFKA, HIVE, FLINK) 
AND issuetype = Epic 
AND due >= -60d AND due <= now() 
AND status not in (Closed, Resolved) 
AND fixVersion in unreleasedVersions() 
AND updated <= -10d 
AND issueFunction in subtasksOf("status not in (Closed, Resolved)") 
ORDER BY due ASC, updated ASC, priority DESC

/* 483. Show all infrastructure and environment related bugs 
in KAFKA and FLINK that were created during business hours in the last 30 days, 
are still unresolved, have no assignee, are linked to at least one open incident, 
and have a severity of critical or blocker — 
identifying production incidents with no owner that are silently ageing */
project in (KAFKA, FLINK) 
AND issuetype = Bug 
AND labels in (infrastructure, environment, devops, incident, production) 
AND priority in (Blocker, Critical) 
AND created >= -30d 
AND created >= startOfDay("-30d") 
AND assignee is EMPTY 
AND status not in (Closed, Resolved) 
AND issueFunction in linkedIssuesOf("issuetype = Incident AND status not in (Closed, Resolved)") 
ORDER BY created ASC, priority DESC

/* 484. Show all issues across KAFKA, HIVE, FLINK, ZOOKEEPER 
that were resolved in the current quarter but were reopened within 7 days of resolution, 
are now assigned to a different person than who originally resolved them, 
and are blocker or critical priority — 
revealing poor first-time fix rate and ownership handoff problems in the team */
project in (KAFKA, HIVE, FLINK, ZOOKEEPER) 
AND issuetype = Bug 
AND status = Reopened 
AND priority in (Blocker, Critical) 
AND resolutiondate >= startOfYear() 
AND resolutiondate <= now() 
AND updated >= -7d 
AND assignee != reporter 
ORDER BY resolutiondate ASC, priority DESC, updated DESC

/* 485. Show all new features and improvements committed 
across KAFKA, HIVE, FLINK for fix versions due to release in the next 30 days, 
that have been in progress for more than 45 days without resolution, 
are not linked to any design or specification issue, 
have no fix version release date set, 
and are assigned to someone who has not updated the issue in over 5 days — 
representing product commitments at highest risk of slipping the roadmap */
project in (KAFKA, HIVE, FLINK) 
AND issuetype in (Improvement, "New Feature") 
AND status = "In Progress" 
AND created <= -45d 
AND updated <= -5d 
AND fixVersion in unreleasedVersions() 
AND due <= 30d 
AND issueFunction not in linkedIssuesOf("issuetype in (Specification, Design, Proposal)") 
AND assignee is not EMPTY 
ORDER BY due ASC, created ASC, priority DESC


