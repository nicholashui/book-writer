---
topic: two-book-overlap
title: Two Book Overlap
language: en
register: null
source_files:
  - testdata/two-book-overlap/book-a.md
  - testdata/two-book-overlap/book-b.md
generated_at: <stamp>
model: grok-4
combiner_version: 0.1.0
---

# Two Book Overlap

## Contents

- Cues
  - Signals
    - Section
      - Section
      - Section
      - Section
    - Section
- Appendix A — Sources
- Appendix B — Quizzes
  - Section Section
- Appendix E — Conflicts
  - c-s-eyes-1

## Cues

### Signals

#### Section

##### Section

(眼睛是心灵的窗口)。(目光方向可以透露一个人的兴趣与注意力所在)。(瞳孔放大通常表示兴趣)，(频繁眨眼则可能表示紧张不安)。

##### Section

(眼睛是心灵的窗口)。(瞳孔放大表示兴趣)。(眨眼频率变化也能透露紧张)。(视线回避可能表示不安)，(或者对方正在思考下一步)。

##### Section

(瞳孔在强光下会缩小)，(并不是兴趣的可靠信号)。

(瞳孔放大表示兴趣)，(而且不会在强光下缩小)。 [book-b#c-s-eyes-1]

#### Section

(握手时掌心出汗属于精神性出汗)，(而不是温热性出汗)。(用力过猛的握手可能表示支配欲)，(而不是礼貌客气)。

## Appendix A — Sources

| stem | bytes | sha256 |
|---|---:|---|
| book-a | <bytes> | <sha256> |
| book-b | <bytes> | <sha256> |

## Appendix B — Quizzes

### Section Section

(每个人都有自己的情绪)。(那么你的情绪是稳定的吗)？

1. (每天清晨起床时)，(你经常有什么样的感觉)？
   A. (忧郁)
   B. (快乐)
   C. (说不清楚)

(结果分析)：(总分越高说明情绪越不稳定)。

## Appendix E — Conflicts

### c-s-eyes-1

- book-b: (瞳孔在强光下会缩小)，(并不是兴趣的可靠信号)。
- book-a: (瞳孔放大表示兴趣)，(而且不会在强光下缩小)。
- inline_tag: `[book-b#c-s-eyes-1]`
