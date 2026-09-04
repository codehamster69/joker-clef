# Speaker Script — Process-Aware Authenticity Detection in Supervised Programming Labs

A slide-by-slide talking script for `Process-Aware_Authenticity_Detection_Presentation.pptx` (13 slides).
Each section gives: the slide's on-screen content (for reference), suggested spoken delivery, approximate
timing, and a transition line into the next slide. Total runtime target: **~12–15 minutes**, suitable for
an M.Tech proposal defense or committee presentation, with room to compress for a shorter slot.

---

## Slide 1 — Title
**On screen:** "Process-Aware Authenticity Detection in Supervised Programming Labs" · subtitle · author line.

**Say:**
> "Good [morning/afternoon]. My proposal is about a gap that sounds almost too simple to be a research
> problem: in a supervised programming lab, we verify *who is in the room* very well — but we don't verify
> *who is actually typing the code* at all. I'm going to walk through why that gap exists, what the
> literature currently covers, and the study I'm proposing to close it."

**Timing:** ~30 seconds.
**Transition:** "Let's start with the problem itself."

---

## Slide 2 — The Problem
**On screen:** "Verified today" vs. "Not verified" comparison; the two-sentence framing at the bottom.

**Say:**
> "Every lab has some identity check at the door — an ID card, a face match, a biometric scan, an RFID
> tap. That tells you the enrolled student walked in. What it does not tell you is who is producing the
> code on their screen for the next ninety minutes. A friend, a senior, or a hired proxy can sit down at
> that same terminal — after the identity check has already passed — and type the entire solution live,
> in full view of the instructor. No plagiarism detector catches this, because the code is original. No
> AI-detector catches this, because it wasn't written by an LLM. It was written by the wrong human, live,
> under supervision that was never built to notice."

**Timing:** ~60–75 seconds.
**Transition:** "So who *is* looking at this problem? I looked at two active research literatures — and neither one covers it."

---

## Slide 3 — Literature Landscape
**On screen:** Two side-by-side threads: Process-Driven Authenticity Detection vs. Physical Proxy Attendance Detection.

**Say:**
> "The first thread is process-driven authenticity detection — keystroke timing, edit sequences — but it's
> built for *take-home* cheating. The adversary is a student alone, at home, with unlimited time to disguise
> an LLM answer. The closest system here, Nosi IDE, published just five months before this proposal, at a
> top venue — I'll come back to that.
>
> The second thread is physical proxy attendance detection — RFID, face match, weight sensors. This solves
> a real problem: someone else checking in *for* you. But it verifies identity at a single checkpoint, and
> has nothing to say about the rest of the session.
>
> Notice the punch line at the bottom: neither of these is built for a student who is physically present
> and identity-verified, while someone else does the actual coding, live, under supervision."

**Timing:** ~75–90 seconds.
**Transition:** "That gap is worth drawing out explicitly, because it's the whole thesis in one picture."

---

## Slide 4 — Research Gap (flowchart)
**On screen:** The two-thread convergence diagram funneling into the amber "Unaddressed Gap" box.

**Say:**
> "This is that picture. Take-home process detection on the left, checkpoint attendance detection on the
> right — both active, both publishing recent work, both converging on the same blind spot: live, in-person
> authorship verification inside a supervised session. That's the space this proposal sits in."

**Timing:** ~30–40 seconds. *(Let the diagram do the work — don't over-narrate it.)*
**Transition:** "Before I propose anything, it's worth being precise about what's already established versus what's genuinely open."

---

## Slide 5 — Confirmed vs. Open
**On screen:** Four rows — two confirmed claims, one open application, one clearly missing.

**Say:**
> "CodeBench already proved fine-grained keystroke data can be collected at scale — over two thousand
> students, sixteen courses. Sequential and clustering analysis of coding trajectories is also well
> established for predicting outcomes. Those are confirmed.
>
> What's *not* confirmed — and this is a narrower, more honest claim than 'nobody has done anything like
> this' — is using AST structural-jump detection as a *live, single-session* anomaly signal. Clone-detection
> tools like CP-Miner exist, but they compare files *across* a codebase offline, not one file evolving in
> real time. And, as expected, nobody in the proxy-attendance literature uses IDE process data at all —
> which confirms the gap rather than just asserting it."

**Timing:** ~60 seconds.
**Transition:** "So how does the proposed study actually compare, side by side, to the two existing threads?"

---

## Slide 6 — Comparative Analysis
**On screen:** Four-column table — dimension, process-driven work, attendance systems, proposed study.

**Say:**
> "Walking down the rows: the adversary model shifts from 'solo student, unlimited time' to 'someone else
> physically at your terminal, live.' The setting shifts from take-home or a single checkpoint, to a fully
> supervised in-session lab. The signal set keeps keystroke timing and edit sequences from prior work, but
> adds AST structural jumps. And critically, the ground truth shifts from a five-person proof-of-concept
> to controlled, labeled in-lab groups — genuine, proxy-typed, and minimal-interaction. That last row is
> the methodological upgrade this thesis is actually built on."

**Timing:** ~60–75 seconds.
**Transition:** "All of that comparison collapses into one specific, testable research question."

---

## Slide 7 — Research Question
**On screen:** The boxed research question; supporting note on positioning against Nosi IDE.

**Say:**
> "Can IDE-level process signals — keystroke timing, edit and execution sequences, and structural code
> jumps — distinguish a student's own authentic in-lab work from proxy-typed or minimally-engaged work? And
> do the *same* signals that already work for take-home LLM-scribing detection generalize to this live,
> in-person setting, or do they break down?
>
> I want to be direct about why this is framed as a two-part question. It positions this thesis explicitly
> against Lueder et al.'s Nosi IDE, published in 2026 — rather than re-presenting 'watch the process, not
> the output' as if it were a new idea. And both possible answers are useful: if the signals transfer, that
> strengthens the case for a unified authenticity framework. If they don't, that's an equally real finding
> about how setting-specific the field's leading detection signal actually is."

**Timing:** ~75–90 seconds. *(This is the thesis's center of gravity — don't rush it.)*
**Transition:** "To answer that question, here's the system I'm proposing to build."

---

## Slide 8 — System Architecture (flowchart)
**On screen:** Four-stage pipeline — Event Collector → Feature Extraction → Authenticity Classifier → Instructor Dashboard.

**Say:**
> "Four stages. The event collector captures keystrokes, pastes, runs, and edits with timestamps — this
> part is deliberately unoriginal; Nosi IDE and CodeBench already do this well. Feature extraction computes
> keystroke inter-arrival time and edit-sequence shape — both carried over from prior work — plus AST-diff
> structural jumps, which is the new feature family motivated by the gap on slide five. The classifier is a
> three-way split — genuine, proxy-typed, minimal-interaction — not binary, because pasting your own
> boilerplate and having someone else type for you are different failure modes with different signatures.
> And the dashboard surfaces a flag *with an evidence trace*, not just a score, so an instructor can actually
> act on it.
>
> The key design choice is at the bottom: this runs on lab workstations already under instructor
> supervision — which is what makes a controlled, multi-group study possible, instead of another small
> take-home proof-of-concept."

**Timing:** ~75–90 seconds.
**Transition:** "That supervision context is exactly what the study design exploits."

---

## Slide 9 — Study Design (flowchart)
**On screen:** Groups A/B/C converging into instrumented sessions, branching into three validation layers, converging into evaluation.

**Say:**
> "Three groups, each with a known ground-truth label at collection time. Group A is genuine — the student
> solves it themselves. Group B is proxy-typed — a second person actually types the solution live at the
> student's terminal, reproducing the real adversary rather than simulating it after the fact. Group C is
> minimal-interaction — pasting a near-complete solution with little iteration. I included Group C
> specifically because it's easy to confuse with proxy-typing if the feature set isn't discriminating
> enough — it's a deliberate stress test on the classifier.
>
> All three groups feed into fully logged, instrumented sessions, which then get validated three ways:
> against the controlled labels themselves, against independent instructor labeling done blind to group
> assignment, and against a held-out retest — can the student solve a similar problem on their own
> afterward? That last layer ties the authenticity signal back to something educationally meaningful,
> not just a detection score."

**Timing:** ~75–90 seconds.
**Transition:** "Two things I want to be upfront about before the roadmap: ethics, and scope."

---

## Slide 10 — Ethics & Scope Discipline
**On screen:** Two panels — Ethics Note, Scope Discipline.

**Say:**
> "Keystroke logs can act as a biometric identifier in their own right, independent of what's typed — so
> this requires a de-identification plan and IRB approval before any deployment, and the same data-sharing
> safeguards already established in prior keystroke-data surveys.
>
> On scope: the original pitch behind this work suggested five parallel research directions — engagement
> scoring, authenticity detection, outcome prediction, AST fingerprinting, a full intervention loop — across
> a hundred to three hundred students. That's multiple papers, not one thesis. I've deliberately scoped this
> down to one defensible contribution — the controlled authenticity-classification study — and labeled the
> rest as future work, not promised deliverables."

**Timing:** ~60–75 seconds.
**Transition:** "Pulling all of that together, here's the shape of the thesis end to end."

---

## Slide 11 — Research Roadmap (flowchart)
**On screen:** Horizontal six-stage flow — Problem → Literature Gap → Research Question → System + Study Design → Validation → Expected Contributions.

**Say:**
> "This is the whole narrative in one line: the problem labs don't verify authorship of live work; the
> literature gap between take-home detection and checkpoint attendance systems; the research question that
> gap produces; the system and controlled study I'd build to answer it; the three-layer validation; and the
> contributions that follow. Everything on the earlier slides maps onto one of these six boxes."

**Timing:** ~30–40 seconds. *(Again, let the diagram carry it — this slide should feel like a recap, not new information.)*
**Transition:** "So, concretely, what does this thesis contribute?"

---

## Slide 12 — Expected Contributions
**On screen:** Four numbered contribution statements.

**Say:**
> "First, the first controlled comparison of take-home authenticity signals against a live, in-person
> proxy-typing adversary — directly testing whether Nosi IDE's result generalizes.
> Second, a ground-truth methodology built on in-lab controlled groups, which is reusable beyond this
> specific thesis.
> Third, an empirical answer — positive or null — to whether existing process features transfer across
> adversary settings. I want to stress that a null result here is not a failure mode for the thesis; it's a
> genuine boundary-condition finding about the field's leading signal.
> And fourth, a scoped, deployable instructor-facing flagging tool built specifically for supervised labs,
> distinct from Nosi IDE's take-home use case."

**Timing:** ~60 seconds.
**Transition:** "That's the proposal. I'll close there and take questions."

---

## Slide 13 — Thank You / Questions
**On screen:** "Thank You" / "Questions & Discussion".

**Say:**
> "Thank you — I'm happy to take questions, and I'm glad to go deeper into the feature engineering, the
> study design, or the ethics/IRB plan, whichever is most useful."

**Timing:** ~15 seconds, then open floor.

---

## Anticipated Questions & Suggested Answers

**Q: How is this different from just... watching the student on CCTV?**
> CCTV verifies presence and gross physical behavior, not what's happening at the keyboard — it can't
> distinguish confident authentic typing from a well-rehearsed proxy typing quickly, and it doesn't scale
> to reviewing footage for an entire lab section. This is a process-level signal, not a visual one, and it's
> designed to produce an evidence trace an instructor can inspect in seconds rather than footage they'd
> have to watch.

**Q: Won't Nosi IDE or someone else just publish this exact live-lab extension first?**
> Possibly — the field is moving fast, and 2026 is clearly the most active year in this space. But that's
> evidence the problem is real, not a reason to abandon it. The differentiator is the controlled,
> multi-group ground-truth methodology (Groups A/B/C) — even if the application target converges with
> someone else's work, the methodology is the more durable contribution.

**Q: How do you know your "proxy" participants in the controlled study behave like a real, covert proxy would?**
> That's an honest limitation, flagged directly in the risks section. Volunteers who know they're being
> studied may not perfectly replicate a covert proxy's behavior. The mitigation is recruiting proxies with
> varying familiarity with the material, and treating this as a documented external-validity limitation
> rather than an unaddressed threat.

**Q: What happens if the signals don't transfer from take-home to live-lab at all?**
> That's still a publishable, valuable result — it would mean the field's leading detection signal is more
> setting-specific than currently assumed, and that live supervised environments need their own feature
> engineering rather than borrowing take-home features wholesale. The thesis is designed so either outcome
> is a contribution, not a dead end.

**Q: Why three classes (genuine / proxy-typed / minimal-interaction) instead of a simple binary "authentic vs. not"?**
> Because those are different failure modes with likely different statistical signatures. A student
> pasting their own boilerplate is not being proxy-typed, but might trigger similar structural-jump signals
> if the classifier isn't discriminating enough — collapsing them into one "not authentic" label would
> hide exactly the distinction the classifier needs to get right, and would also tell an instructor less
> about what actually happened.
