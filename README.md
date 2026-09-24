# Decision Models Under Pressure

Seven systems do the same job: take a piece of text, a question, and a list of
candidate answers, and return a probability over those candidates. I measured
them against each other as that job gets harder in the three ways it gets harder
in production. The candidate list grows, the option order changes, and the wrong
answers stop being obvious.

**Short version.** TypeSafe's
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) held up
best as the list grew and best when the wrong answers got plausible. Shuffle the
option order, though, and it changes its answer on one decision in seven. Two of
the open models never change theirs, because they cannot.

| at 64 to 128 candidates | Jev | [Laya](https://huggingface.co/convaiinnovations/laya) | best open |
| --- | --- | --- | --- |
| accuracy at 128 candidates | **60%** | 39% | 41% |
| accuracy lost per doubling of the list | **-0.043** | -0.073 | -0.048 |
| accuracy lost when distractors get hard | **-0.105** | -0.351 | -0.256 |
| answers changed by reordering alone | 14.6% | 49.4% | **0.0%** |

Jev's makers call it a new class of model. Laya's author has said publicly that
he published the same idea a year earlier. On the first claim the numbers are
not kind to the marketing, and on the second they are not kind to Laya: the idea
does look older than Jev, and Jev is still the better implementation of it by a
wide margin.

The other five models are open ones I added so those two numbers would mean
something. Most comparisons of these systems report one accuracy number at one
candidate-set size, which hides most of what matters, because the ranking
changes depending on how many options you offer.

Everything ran on one frozen dataset, with the comparisons and the pass/fail
rules written down before the first call ([PLAN.md](PLAN.md)). n is 200 items per
domain, which is enough to separate the large effects below and not enough for
the small ones. Every caveat is load-bearing.

## What came out of it

Every model gets worse as the candidate list grows. The difference is how fast.

![Accuracy against candidate-set size](docs/figures/k-curve.png)

Jev starts highest and stays highest. At 128 candidates it answers 60%
correctly, where Laya manages 39% and the best open model 41%. Chance at that
list length is 0.8%, so everything here is doing real work; the question is how
much of it survives a longer list.

Its decline per doubling of the list is the shallowest of the seven, shallower
even than the embedding scorers whose design is supposed to help them scale.
Measured over the same range for every model (K=2 to 128, since Jev's API
refuses more than 255 options):

| model | accuracy lost per doubling of the candidate list |
| --- | --- |
| Jev | **-0.043** |
| gte-large | -0.048 |
| bge-large | -0.051 |
| gliclass | -0.064 |
| deberta-large | -0.070 |
| deberta-base | -0.070 |
| Laya | -0.073 |

Notice how little a single-number benchmark would tell you. At two candidates
Laya sits second of seven and trails Jev by two points. At 128 it has fallen to
fourth and trails by twenty-two.

Next the wrong answers had to get harder. Every item exists in two versions: one
where the distractors come from unrelated domains, and one where they are the
right answer's nearest neighbours, so `create alarm` competes against
`delete alarm` and `snooze alarm` rather than against `musical work`. I matched
the surface shape of the words between the two versions so only meaning
separates them. The gap between them is how much a model was leaning on the
wrong answers being obvious.

![Accuracy lost when distractors are nearly right](docs/figures/near-distractors.png)

At 64 candidates Jev drops from 96.8% to 86.3%, losing about a tenth of what it
had. Laya drops from 90.5% to 56.0% and gliclass from 87.0% to 51.2%, each
losing something closer to four tenths. That is the widest spread in the whole
experiment and the one I'd care most about in production, where candidate sets
are full of near misses.

Then I shuffled the options. Same question, same candidates, five different
orderings, and I counted how often the answer changed.

This is where it helps to know how each model reads its candidate list, because
that single design choice predicts the result almost perfectly:

| model | how it reads the candidates |
| --- | --- |
| Jev | undisclosed, but its answers depend on the order, so not one at a time |
| Laya | all candidates in one pass, as a set |
| `gliclass-large-v3.0` | the same |
| `deberta-v3-base-zeroshot-v2.0` | one pass per candidate, each scored against the text alone |
| `deberta-v3-large-zeroshot-v2.0` | the same, larger |
| `bge-large-en-v1.5` | embeds the text once, compares it to cached candidate vectors |
| `thenlper/gte-large` | the same, different encoder |

A model that scores each candidate on its own cannot notice that two candidates
are similar, and the order you list them in cannot reach it. A model that reads
them as a set gets the first ability and the second problem together.

![Answers that change when only the order changes](docs/figures/order-flips.png)

Two of the four candidate-at-a-time models never flip: `bge-large` and
`deberta-large` are exactly 0.0000 across 1,600 decisions each, which is what
scoring each option in isolation should give you. It doubles as a check that the
harness isn't shuffling something it shouldn't.

The other two are not quite zero. `deberta-base` flips on 0.2% of decisions and
`gte-large` on 2.0%, and both turned out to be exact scoring ties broken by
position rather than a real order effect: all 32 of gte's flipping items at 64
candidates have two options tied to within 1e-9. Structural invariance holds. Ties
are the crack in it, and if you are picking a model because order cannot reach
it, 2% is still 2%.

Against that baseline, Jev's 14.6% at 64 candidates is a real cost. Roughly one
decision in seven is settled by list position rather than by content. It is also
about a third of Laya's rate, which changes its answer on half of its decisions
at the same list length.

The flips are not spread evenly, and the spread matters more than the headline:

| where | Jev flips, unrelated distractors | Jev flips, near-neighbour distractors |
| --- | --- | --- |
| request routing (clinc) | 1.5% | 9.0% |
| request routing (mtop) | 3.0% | 12.5% |
| financial topics | 5.0% | 22.0% |
| emotional tone | 30.5% | 33.0% |

Intent routing with easy distractors is the best case and it is genuinely stable.
Intent routing with plausible competing options, which is what a real router
faces, runs four to six times worse. Anything subjective is worse again, and
barely improves when the distractors get easy.

Two fixes are available to anyone deploying these today. Present the options in a
fixed canonical order, which at least makes the instability deterministic. Or ask
the same question under several orderings and average, which buys exact stability
at the price of several calls per decision.

One more thing fell out of the logs after the fact. Confidence and correctness
come apart badly as the list grows, and not for everyone:

| model | calibration error at K=2 | at K=256 |
| --- | --- | --- |
| deberta-large | 0.064 | 0.070 |
| deberta-base | 0.066 | 0.075 |
| bge-large | 0.099 | 0.197 |
| gte-large | 0.048 | 0.299 |
| gliclass | 0.150 | 0.327 |
| Laya | 0.032 | **0.573** |

Laya is the best-calibrated model in the set at two candidates and by far the
worst at 256. If you are gating on a confidence threshold, that is the number
that decides whether the gate works. Jev is missing from this table because its
API rounds probabilities to two decimals, which is too coarse to measure
calibration; 98% of its responses contain at least one option at exactly 0.00.

## What this cannot tell you

Jev's training data is not disclosed. The items come from public datasets, so
"trained better" and "has seen these before" cannot be told apart here. The one
hint available points both ways: Jev is steadiest on exactly the intent-style
domains a decision product would plausibly be trained on, and shakiest on the
domain furthest from that. Nothing here settles it, and no outside test can
settle it while the training data stays private.

The Jev comparison was also not part of the original plan. I wrote the
pre-registration around the three open architecture families and listed Jev as a
conditional extra if I got API access. I did, it went in, and it reversed the
conclusion I had been heading toward, which was that reading candidates as a set
carries costs and buys nothing. Jev reads candidates as a set and carries much
smaller costs. So the honest version is that the costs belong to those two open
checkpoints rather than to the architecture. I've left the original reasoning in
[EXPERIMENTS.md](EXPERIMENTS.md) rather than quietly rewriting it.

Three smaller limits. The near-distractor comparison rests on two domains, both
of which every model here has plausibly seen. Jev's API caps candidate lists at
255 options, so the head-to-head stops at 128; the open models run to 256 and
those numbers are in `results/published/`. And n is 200 items per domain, with
bootstrap intervals on everything in EXPERIMENTS.md, which is enough to separate
the large effects here and not enough for anything subtle.

## Reproducing it

```bash
python -m venv .venv
.venv/bin/pip install torch transformers datasets sentence-transformers scikit-learn

.venv/bin/python -m harness.run.local --rq rq3    # run one question
.venv/bin/python -m harness.analyze.rq3
```

The dataset ships complete, so there is no rebuild step. To rebuild it from the
upstream sources anyway:

```bash
.venv/bin/python -m harness.build --n 200         # build and run the checks
.venv/bin/python -m harness.build.export
```

The Jev arm cost $1.04 for the full grid of 29,600 calls:

```bash
export OPENROUTER_API_KEY=sk-or-...
.venv/bin/python -m harness.run.jev --rq rq3 --cap 2.00
```

## Data

`dataset/v3/` holds 1,000 items across five domains, with the item text, the
gold label, both distractor pools and every derived field. It is published under
**CC BY-SA 4.0**, because MTOP is CC BY-SA 4.0 and ShareAlike carries over. If
you reuse it, your version inherits that too. The code is MIT.

| domain | source | licence |
| --- | --- | --- |
| clinc | [CLINC-150](https://huggingface.co/datasets/clinc/clinc_oos) | CC BY 3.0 |
| goemotions | [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Apache 2.0 |
| mtop | [MTOP](https://huggingface.co/datasets/WillHeld/mtop) | CC BY-SA 4.0 |
| dbpedia | [DBpedia Classes](https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes) | CC0 1.0 |
| fintopic | [Twitter Financial News](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic) | MIT |

Full attribution, the upstream papers, what I changed, and the GoEmotions
content warning are in [`dataset/v3/README.md`](dataset/v3/README.md). Texts are
verbatim; every item carries a SHA-256 of its text so you can check it against
the original source. If you use this data, cite the upstream papers, not this
repo.

## What is in here

| path | contents |
| --- | --- |
| `dataset/v3/` | 1,000 items, the 902-option label universe, per-question splits, the checks, datacard |
| `harness/build/` | how the dataset is made: label universe, items, distractor tiers, the eight checks |
| `harness/models/` | model adapters, one per family, plus the hosted Jev client |
| `harness/run/` | the two runners, local and hosted |
| `harness/analyze/` | one analysis per question, plus the figures |
| `results/published/` | the aggregates every table and figure is built from |
| `PLAN.md` | what I decided to measure, and the pass/fail rules, before anything ran |
| `EXPERIMENTS.md` | every run, including the ones that went against what I expected |

## How the dataset was checked

An earlier version of this produced confident results that did not survive
review. Auditing it turned up three things that mattered: a classifier with no
access to the item text could identify the right answer from label formatting
alone, an option's position was a function of its label rather than the item, and
one model's apparent context ceiling was an artifact of how my harness passed
options to it.

I rebuilt the dataset around those failures, and the checks that caught them now
run as assertions before any model does. Eight of them: no option string may
appear in the prompt's own wording, surface features must not separate near
distractors from far ones, a text-blind classifier must stay under twice chance,
gold position must depend on the item, the near tier must be measurably nearer,
every item must be usable in both tiers at every list size, no adjudicated
ambiguous pair may sit together, and each option set must be well formed.

Seven of the eight pass. The one that does not is the text-blind classifier,
which still beats chance on some cells: 0.260 on DBpedia against a chance rate of
0.0625 and a 0.125 gate. That means some accuracy on those cells comes from label
formatting rather than from reading the text, and it inflates the level of the
curves without changing their shape. Per-cell numbers are in
`dataset/v3/gates.json` rather than described away. Two fixes were tried and
rejected, with their numbers recorded: coarser matching reopens the formatting
check, and dropping the affected items shrinks the sample while making the
residual worse.

Two rules I'd written in advance fired against me during the run. One voided a
family-level comparison when the two models inside a family disagreed more than
the families did. The other required the order-invariant models to score exactly
zero, which two of four did, with the other two traced to scoring ties.

## Licence

The dataset in `dataset/v3/` is CC BY-SA 4.0, inherited from MTOP. Per-source
licences, credits and what I changed are in
[`dataset/v3/README.md`](dataset/v3/README.md).

The code carries no licence, which under default copyright means you may read it
but not reuse it. That is deliberate rather than an oversight: the point of this
repo is the method and the numbers, both of which you are welcome to take. If
you want to reuse the code itself, ask and I'll add a licence.
