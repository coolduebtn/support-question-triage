# Failure analysis (dev split, saved runs, no API calls)

Runs: A `20261101-201233-caa31fb7-dev`, A+ `20261101-201235-dbb65c21-dev`, B `20261101-201238-a0ce2387-dev`


## A: what it predicted for each correct action

| correct \ predicted | answer | clarify | escalate | error |
|---|---|---|---|---|
| answer | 15 | 0 | 0 | 0 |
| clarify | 13 | 0 | 0 | 0 |
| escalate | 24 | 0 | 0 | 0 |

## A+: what it predicted for each correct action

| correct \ predicted | answer | clarify | escalate | error |
|---|---|---|---|---|
| answer | 9 | 0 | 6 | 0 |
| clarify | 4 | 0 | 9 | 0 |
| escalate | 13 | 0 | 11 | 0 |

## B: what it predicted for each correct action

| correct \ predicted | answer | clarify | escalate | error |
|---|---|---|---|---|
| answer | 12 | 0 | 3 | 0 |
| clarify | 7 | 5 | 1 | 0 |
| escalate | 4 | 1 | 19 | 0 |

## Per-class recall (share of items of that class the approach got right)

| approach | answer | escalate | clarify (all) | clarify, real | clarify, constructed |
|---|---|---|---|---|---|
| A | 15 of 15 | 0 of 24 | 0 of 13 | 0 of 1 | 0 of 12 |
| A+ | 9 of 15 | 11 of 24 | 0 of 13 | 0 of 1 | 0 of 12 |
| B | 12 of 15 | 19 of 24 | 5 of 13 | 1 of 1 | 4 of 12 |

## B: 16 wrong of 52, by automatic cause hint

| cause hint | count |
|---|---|
| missed_ambiguity | 7 |
| unsafe_answer | 4 |
| class_boundary | 3 |
| retrieval_miss | 2 |

Hints are automatic (retrieval_miss = no gold page in the retrieved pages; missed_ambiguity = answered a CLARIFY item; unsafe_answer = answered an ESCALATE item; class_boundary = any other mismatch). Confirm or correct them by reading the rows.

| qid | correct | predicted | hint | question | why B chose that |
|---|---|---|---|---|---|
| q_4rlrf487n0 | answer | escalate | class_boundary | OFBiz Sanity Test Documentation for Order Entry and Fulfillment  Hello Devs,  I am currently working on creati | The provided excerpts are high-level checklist entries and do not contain the detailed steps, expected results, field names, or business rul |
| q_60ochoozmt | escalate | answer | unsafe_answer | security issue: democustomer can edit login info of admin  hi ofbiz users,  I can repeatedly reproduce the iss | Documentation contains guidance showing this is expected with demo data and how to secure/mitigate (change or disable demo accounts, use Par |
| q_cxzozcvz1n-probe | clarify | answer | missed_ambiguity | Sanity check: OFBiz as a 3PL WMS  Hi all,  I've configured OFBiz as the WMS for a 3PL:  To stop OFBiz from bil | Documentation excerpts contain guidance on both suppressing order-to-cash and the tenancy model for 3PL use. |
| q_d8nv7zb2s2 | escalate | answer | unsafe_answer | Core Apache OFBiz Manufacturing Workflow Diagrams  Hello everyone,  I'd like to highlight a valuable contribut | Excerpts include wiki pages that document manufacturing workflows, diagrams, and related demo data and tools. |
| q_drjvjlxn5c | answer | escalate | retrieval_miss | 回复：Re: ofibz causes system extremely slow after one night  hi [name],  You mean framework\base\config\cache.pr | The provided excerpts do not document cache.properties or how to comment/increase cache settings; they therefore don't cover whether to edit |
| q_hg8vzp02nh | escalate | answer | unsafe_answer | 回复：Re: how to enable login account  hi [name],  Thanks for your reply, again.  Yes, I have read various docs,  | The excerpts explain how to enable a user by assigning security groups via the Party application and describe demo/seed data and default dem |
| q_hhbljjy6mb | escalate | answer | unsafe_answer | Request for Guidance: Importing Aggregated POS Sales and Financial Transactions into OFBiz  Dear OFBiz Communi | Excerpts provide relevant guidance for inventory issuance and for importing/handling financial transactions (FinAccount / bank statement app |
| q_mffxrrfxtr-probe | clarify | answer | missed_ambiguity | Hi,      I am new to ofbiz. Can someone please point to the resources that would be helpful in migrating ofbiz | Excerpts contain links and pages relevant to migrating/upgrading OFBiz (Getting Started, tutorials for versions, and Technical Documentation |
| q_nhl71rlpt0 | escalate | clarify | class_boundary | Maximum concurent users in real OFBiz production system  Hi all, Are there any measurements, how many users ca | Documentation indicates performance depends on customizations and hardware and gives page‑hits rule‑of‑thumb, but does not state a maximum c |
| q_o5yxnf1txb | answer | escalate | retrieval_miss | Use of the grid widget  Good morning everyone  I've been trying to use the grid widget and have some questions | The provided excerpts do not include documentation or examples explaining how to configure or use the grid widget, the significance of list- |
| q_o5yxnf1txb-probe | clarify | escalate | class_boundary | Use of the grid widget  Good morning everyone  I've been trying to use the grid widget and have some questions | The provided excerpts do not document the OFBiz grid widget attributes (e.g. paginate-target, list-name) or how pagination is wired to the s |
| q_oo7p41fj1x-probe | clarify | answer | missed_ambiguity | Hello Team,  Greetings!  I knew how to enable webservices. Does Ofbiz exposes API ootb? If yes, Can you share  | Provided excerpts show OFBiz exposes services and includes a REST API component with Swagger docs. |
| q_rqgw8xtl5z-probe | clarify | answer | missed_ambiguity | Hello Community,  Would you please share a data example if you have one or point to where in the code I can un | Documentation excerpts include example data files and where to find code that loads/displays them. |
| q_rwmckbb73n-probe | clarify | answer | missed_ambiguity | I have a few questions about executing services in async mode.  1. Is there a thread pool that limits number o | The provided excerpts describe the job scheduler's thread pool and its configurable attributes, which answer both questions. |
| q_vxf0nt2czh-probe | clarify | answer | missed_ambiguity | How to manage a project  Hi,  I would like to use OFBiz for project management.  If a kind of project exists,  | Documentation excerpts include the Project Management application description and where to find its code and development resources. |
| q_ykg9fnkg3z-probe | clarify | answer | missed_ambiguity | Custom component does not show up on the menu on a multi tenant setup  Hi All  I am very new to Ofbiz. I have  | The excerpts show how components must be assigned/loaded per-tenant and how seed/security data must be installed; they contain the steps nee |

Follow-up targeting (B, correct CLARIFY items judged): 2 of 5 follow-ups name the missing detail.

## Risk-coverage for A (sweep the top vector cosine; this is what A+'s threshold does)

Coverage = share of all questions answered. Risk = share of answered questions that should not have been answered.

| coverage | risk |
|---|---|
| 0.02 | 0.00 |
| 0.10 | 0.60 |
| 0.17 | 0.56 |
| 0.25 | 0.62 |
| 0.33 | 0.59 |
| 0.40 | 0.62 |
| 0.48 | 0.68 |
| 0.56 | 0.66 |
| 0.63 | 0.67 |
| 0.71 | 0.70 |
| 0.79 | 0.73 |
| 0.87 | 0.69 |
| 0.94 | 0.71 |

If risk stays high at every coverage level, similarity alone cannot separate answerable from unanswerable questions.
