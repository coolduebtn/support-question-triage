# Failure analysis (test split, saved runs, no API calls)

Runs: A `20261101-194635-caa31fb7-test`, A+ `20261101-194638-dbb65c21-test`, B `20261101-194740-a0ce2387-test`


## A: what it predicted for each correct action

| correct \ predicted | answer | clarify | escalate | error |
|---|---|---|---|---|
| answer | 13 | 0 | 0 | 0 |
| clarify | 11 | 0 | 0 | 0 |
| escalate | 41 | 0 | 0 | 0 |

## A+: what it predicted for each correct action

| correct \ predicted | answer | clarify | escalate | error |
|---|---|---|---|---|
| answer | 5 | 0 | 8 | 0 |
| clarify | 2 | 0 | 9 | 0 |
| escalate | 11 | 0 | 30 | 0 |

## B: what it predicted for each correct action

| correct \ predicted | answer | clarify | escalate | error |
|---|---|---|---|---|
| answer | 8 | 2 | 3 | 0 |
| clarify | 4 | 6 | 1 | 0 |
| escalate | 6 | 4 | 31 | 0 |

## Per-class recall (share of items of that class the approach got right)

| approach | answer | escalate | clarify (all) | clarify, real | clarify, constructed |
|---|---|---|---|---|---|
| A | 13 of 13 | 0 of 41 | 0 of 11 | 0 of 3 | 0 of 8 |
| A+ | 5 of 13 | 30 of 41 | 0 of 11 | 0 of 3 | 0 of 8 |
| B | 8 of 13 | 31 of 41 | 6 of 11 | 2 of 3 | 4 of 8 |

## B: 20 wrong of 65, by automatic cause hint

| cause hint | count |
|---|---|
| class_boundary | 7 |
| unsafe_answer | 6 |
| missed_ambiguity | 4 |
| retrieval_miss | 3 |

Hints are automatic (retrieval_miss = no gold page in the retrieved pages; missed_ambiguity = answered a CLARIFY item; unsafe_answer = answered an ESCALATE item; class_boundary = any other mismatch). Confirm or correct them by reading the rows.

| qid | correct | predicted | hint | question | why B chose that |
|---|---|---|---|---|---|
| q_17pwk6swl9 | escalate | clarify | class_boundary | Proof of Concept: Modern Angular/Flutter Frontends & REST API Plugin for OFBiz  Hi everyone,  For anyone looki | The provided excerpts discuss modern UIs, REST services and demo/seed data, but they do not state which OFBiz version or environment you are |
| q_2vhtsxh077 | answer | clarify | retrieval_miss | Invoice Date  Hello Community,  Why does the user interface not allow editing the invoice due date? | Documentation shows the Invoice Header screen can update the Due Date, but the reports/screens also mention possible bugs; the excerpts do n |
| q_2vhtsxh077-probe | clarify | answer | missed_ambiguity | Hello Community,  Why does the user interface not allow editing the invoice ? | The provided excerpts explain which invoice/header/item fields are editable, that some item fields cannot be changed and must be removed/rec |
| q_5sq19gm80v | answer | clarify | class_boundary | how to upgrade ofbiz  Hi,  How does one upgrade OFBiz from one release to the other?  Do you clone the repo an | Docs show different update steps and tips for SVN vs Git and mention repository-update commands; I need to know which VCS you use to give th |
| q_7y2tgjclf9 | escalate | answer | unsafe_answer | Issue in OFBiz Demo – "User Authorization Required for executeMrp"  Dear OFBiz Development Team,  I hope you a | Excerpts provide general troubleshooting steps to fix missing permission errors and how to edit user security groups/permissions in OFBiz de |
| q_bxqrcfdtvd | answer | escalate | class_boundary | Invoice status to Paid without actual paymet  Hello Community,  Creating a sales order with COD payment method | The provided excerpts do not explain how an invoice can be marked 'paid' while its associated payment remains 'not paid' (no description of  |
| q_fy79lk14np-probe | clarify | escalate | class_boundary | Confused about service  Hello Community,  In the subject service, there's the following comment and line of co | The provided excerpts do not document the mini-language <set> element or the specific mapping/semantics you’re asking about (why a value is  |
| q_g92sgt6j3m-probe | clarify | answer | missed_ambiguity | What is the difference between and  Hello Community,  Would you please shed some light on this? | Excerpts contain a direct comparison of services vs events. |
| q_hjx80n7kb9 | escalate | clarify | class_boundary | Build failure at test task on Apache Ofbiz 18.12.16 setup  Hi,  I am using JDK 17, with gradle wrapper gradle- | The provided docs indicate tests require test data/delegator setup (must run 'gradlew loadDefault' before running integration tests) and men |
| q_hz0ohskd9k | escalate | clarify | class_boundary | Subject: Help Needed: Unable to Create Commission Run in Accounting Module (OFBiz 18.12.19)  Dear OFBiz Commun | Documentation shows Commission Agreements exist and that billing/automation (which drives invoices/commission processing) only fires for sal |
| q_k0hc0j41cs | escalate | answer | unsafe_answer | Performance scale up Thread  Dear All, related to performance scale up thread wanted to share one observation  | Excerpts provide profiling guidance and heap/GC tuning recommendations sufficient to answer both questions. |
| q_k6jyqq3y6s | escalate | clarify | class_boundary | Demos down  Hi,  I don't know why and don't have time to check, sorry  [name] | The provided excerpts cover demo setup and demo data but to apply them I need to know which environment is affected (public demo servers vs  |
| q_l3fdh8zwy5 | escalate | answer | unsafe_answer | Sender Party and Receiver Party bank details in entity Payment  Hi all, I want to understand how entities: pay | Excerpts show where bank details are stored and how payment method maps to company account. |
| q_m7g9qjsjz5 | clarify | answer | missed_ambiguity | Unable to Complete Order  Hello Comunity, I want to complete an order in the store I created.  In the existing | Excerpts describe QuickShip prerequisites and the Warehouse Fulfillment path which explain the likely causes and next steps. |
| q_pfp4odqbyb | escalate | answer | unsafe_answer | Problem in running the projest for the first time  Hi Community,  I downloaded 24.09.01 and running "gradle cl | OFBiz documentation instructs to use the project Gradle wrapper (gradlew) rather than a system-installed Gradle; that directly addresses the |
| q_rfbj6762c9 | answer | escalate | retrieval_miss | Request for Guidance on Upgrading Customized Apache OFBiz 16 to 18.12.12  Dear Apache OFBiz Team,  Hope this e | The provided excerpts are high-level overviews and release tutorial introductions; they do not include any upgrade or migration procedures,  |
| q_sl8j4qzttd | answer | escalate | retrieval_miss | Negative stock caused by transfer entry errors - root cause and fixes?  Hi all,  We're seeing negative stock b | The provided excerpts do not contain procedural details or documented services to correct negative stock, nor do they describe transfer-entr |
| q_sl8j4qzttd-probe | clarify | answer | missed_ambiguity | Negative stock caused by transfer entry errors - root cause and fixes?  Hi all,  We're seeing negative stock b | Excerpts cover variance/adjustment mechanism, inventory mover/cycle counts, and warehouse stock-move controls needed to answer. |
| q_t8p3f9dgxw | escalate | answer | unsafe_answer | GOing to FOSDEM - any OFBiz devs/users that would like to meet?  Hello,  I'm going to FOSDEM. I would like to  | Documentation shows community meetup practice and gives channels to reach OFBiz users/devs. |
| q_z54voxyz1x | escalate | answer | unsafe_answer | Cost Centers functionality  Hi all,In the system (e.g., the demo trunk), I see the menus GL Account Category ( | Docs state that Cost Centers/GL Account Category are used to split/post amounts by percentage, but the provided excerpts do not show the spe |

Follow-up targeting (B, correct CLARIFY items judged): 1 of 6 follow-ups name the missing detail.

## Risk-coverage for A (sweep the top vector cosine; this is what A+'s threshold does)

Coverage = share of all questions answered. Risk = share of answered questions that should not have been answered.

| coverage | risk |
|---|---|
| 0.02 | 0.00 |
| 0.09 | 0.50 |
| 0.17 | 0.55 |
| 0.25 | 0.69 |
| 0.32 | 0.71 |
| 0.40 | 0.77 |
| 0.48 | 0.68 |
| 0.55 | 0.69 |
| 0.63 | 0.71 |
| 0.71 | 0.74 |
| 0.78 | 0.76 |
| 0.86 | 0.79 |
| 0.94 | 0.80 |

If risk stays high at every coverage level, similarity alone cannot separate answerable from unanswerable questions.
