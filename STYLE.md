# TUI Style Guide

A short visual guide for keeping llamactl NEO consistent.

## Visual language

- Use **cyan** for titles, active elements, case names, and primary emphasis.
- Use **dark gray** for labels, metadata, help text, inactive values, and placeholders.
- Use **green** for healthy state and decode/token throughput.
- Use **yellow** for warnings, prompt-processing throughput, latency, and attention.
- Use **red** only for errors or destructive states.
- Selected rows use a cyan background with black, bold text.

## Cards and modals

- Use rounded borders and the shared `title()` helper.
- Center modals and call `Clear` before rendering them.
- Use one character of horizontal padding inside modal borders.
- Keep modals only as tall as their content requires.
- Keep border titles to one uppercase label; put profile names, controls, and other context in the body or footer.
- Put controls last: `Enter/y confirm - Esc/n cancel`.
- Destructive confirmation text should be yellow and bold, not decorative.

## Tables and metrics

- Prefer tables over manually spaced text when values form columns.
- Keep column meanings explicit: `PP` is prompt processing; `T/S` is decode speed.
- Keep related measurements adjacent, such as `PP-S` and `T/S-S`.
- Always render expected rows; use `--` for unavailable or unfinished values.
- Hide detail responsively instead of squeezing unreadable columns.
- Preserve important live measurements in their own clearly labeled columns.
- Display rates with one decimal place, memory in GiB, and elapsed time in seconds.

## Typography and spacing

- Use uppercase for card titles, table headings, and fixed category names.
- Use bold sparingly for titles, selections, and primary identifiers.
- Prefer concise labels and dash separators: `value - value`.
- Use at most one blank line between related sections.
- Avoid repeating the same metadata in both a heading and the body.

## Interaction

- Lock unrelated controls while a modal task is running.
- Keep cancellation available through `Esc`, with `q`, `c`, or `n` where appropriate.
- Update the footer legend whenever modal controls replace page controls.
- Confirm destructive actions; reversible navigation should remain immediate.
