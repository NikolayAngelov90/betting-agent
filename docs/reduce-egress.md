# Mission

You are a senior Python performance engineer specializing in Supabase, PostgreSQL, async Python, AI agents and network optimization.

This repository is a betting agent written in Python.

Its Supabase project exceeded the monthly Egress quota on the Free Plan.

Your ONLY objective is to reduce Supabase Egress usage as much as possible while keeping identical behaviour.

Do not optimize CPU.
Do not optimize RAM.
Do not optimize code style.

Only optimize everything that causes unnecessary data transfer from Supabase.

---

# First step

DO NOT modify code immediately.

First inspect the ENTIRE repository.

Map every place where Supabase is used.

Create a report containing:

- every Supabase client
- every query
- every insert
- every update
- every upsert
- every delete
- every select
- every RPC
- every Storage request
- every Auth request

Also create a dependency graph showing:

Python job
↓

Supabase query
↓

returned data

↓

consumer

Only after this analysis begin implementing improvements.

---

# Identify the biggest egress sources

Rank every source by estimated bandwidth.

Example:

1. Historical match loading
HIGH

2. Prediction loading
MEDIUM

3. Team statistics
MEDIUM

4. Daily fixtures
LOW

5. Auth refresh
LOW

Focus on the biggest sources first.

---

# Audit every Supabase query

Search for:

select("*")

Replace with explicit columns.

Example

BAD

select("*")

GOOD

select("id,fixture_id,home_team,away_team,status")

Never fetch unused columns.

---

# Detect repeated reads

Find places where identical queries are executed repeatedly.

Especially inside

- loops
- schedulers
- cron jobs
- retries
- polling
- background workers

Cache results whenever safe.

---

# Historical data

Check whether historical data is downloaded repeatedly.

If historical fixtures rarely change:

Store them locally.

Examples:

SQLite

Parquet

Pickle

JSON cache

Disk cache

instead of requesting them every run.

---

# API Football synchronization

Inspect synchronization logic.

Only request:

new matches

updated matches

recent fixtures

Never reload complete datasets.

---

# Incremental sync

Whenever possible:

Download only records modified after the latest timestamp.

Avoid full table scans.

---

# Batch operations

Replace many small Supabase requests with batch operations.

Examples:

Multiple inserts

↓

Single insert

Multiple updates

↓

Batch update

Multiple selects

↓

One query

---

# Avoid read-after-write

Search for patterns like:

insert

↓

select

If the inserted object is already known, remove the extra read.

---

# Local cache

If the same lookup happens frequently:

Store locally.

Examples:

team ids

league ids

season ids

country ids

bookmaker ids

market ids

These should almost never require repeated Supabase reads.

---

# Polling

Inspect scheduled jobs.

Find polling intervals.

If data changes hourly, never poll every minute.

Reduce unnecessary reads.

---

# Async execution

If multiple identical async tasks request the same data:

Deduplicate them.

Share cached results.

---

# Payload size

Reduce returned JSON size.

Never request:

unused text

unused metadata

unused nested objects

large arrays

Only request fields actually consumed.

---

# Pagination

Never fetch an entire table.

Use

range()

limit()

pagination

streaming

where appropriate.

---

# Database

Inspect SQL efficiency.

Find:

missing indexes

slow filters

sequential scans

poor joins

Recommend indexes where useful.

---

# Storage

If Supabase Storage is used:

Check

duplicate downloads

large files

images

model files

CSV files

Cache locally whenever possible.

---

# Logging

Ensure logging never serializes huge Supabase responses.

Avoid printing entire objects.

---

# Retry logic

Inspect retries.

Avoid downloading identical data after temporary failures.

Retry only failed requests.

---

# Configuration

Inspect polling intervals.

Inspect scheduler frequency.

Inspect background jobs.

Determine whether the application is requesting fresh data more often than necessary.

---

# Final implementation

Implement optimizations in order of highest expected bandwidth reduction.

Do not introduce breaking changes.

Preserve behaviour.

---

# Final report

For every optimization provide:

File

Problem

Why it causes egress

Estimated impact

Code changes

Expected bandwidth reduction

---

# Final summary

Include:

- Top 10 biggest egress sources
- Estimated percentage reduction
- Remaining opportunities
- Files modified
- Any trade-offs