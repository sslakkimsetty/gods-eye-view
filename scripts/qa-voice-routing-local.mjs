#!/usr/bin/env node
/**
 * qa-voice-routing-local.mjs — routing eval against a local OpenAI-compatible server.
 *
 * The sibling harness (qa-voice-routing.mjs, LAYER 1) asserts tool routing through a
 * real OpenAI Realtime WebSocket session. That transport does not exist for a local
 * model, so this script asks the same question over plain chat-completions:
 *
 *   given the PRODUCTION instructions and tool definitions, does the model call the
 *   right tool for each phrase?
 *
 * It deliberately shares both inputs with production rather than copying them —
 * GEV_REALTIME_INSTRUCTIONS and GEV_REALTIME_TOOLS come from vite.config.js, and the
 * phrase table comes from voiceRoutingPhrases.mjs — so this cannot silently drift from
 * what the app actually ships.
 *
 * Usage:
 *   node scripts/qa-voice-routing-local.mjs
 *   node scripts/qa-voice-routing-local.mjs --model mlx-community/Qwen3.6-35B-A3B-6bit
 *   node scripts/qa-voice-routing-local.mjs --prompt full|slim|local
 *       full  = the Realtime prompt, slim = a 4-line control, local = what ships
 *   node scripts/qa-voice-routing-local.mjs --only Tokyo       # substring filter
 *   node scripts/qa-voice-routing-local.mjs --jsonl runs/local.jsonl
 *
 * Exit code is 1 when any phrase fails, so this is CI-usable.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  GEV_LOCAL_VOICE_INSTRUCTIONS,
  GEV_REALTIME_INSTRUCTIONS,
  GEV_REALTIME_TOOLS,
} from '../vite.config.js';
import { PHRASES } from './voiceRoutingPhrases.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── CLI ─────────────────────────────────────────────────────
function getOpt(flag, fallback) {
  const i = process.argv.indexOf(flag);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const BASE_URL = getOpt('--url', process.env.GEV_VOICE_LLM_BASE_URL || 'http://127.0.0.1:11452/v1');
const MODEL = getOpt('--model', process.env.GEV_VOICE_LLM_MODEL || 'mlx-community/Qwen3.6-35B-A3B-6bit');
const PROMPT_MODE = getOpt('--prompt', 'local'); // full | slim | local
const ONLY = getOpt('--only', null);
const JSONL = getOpt('--jsonl', null);
const MAX_TOKENS = Number(getOpt('--max-tokens', '600'));
const TIMEOUT_MS = Number(getOpt('--timeout', '120000'));

/**
 * Trimmed system prompt.
 *
 * The production prompt is ~18.6k characters of behavioural rules tuned against the
 * OpenAI Realtime model. Running `--prompt slim` alongside `--prompt full` separates
 * "the model cannot route" from "the model cannot follow this specific prompt".
 */
const SLIM_INSTRUCTIONS = [
  "You control God's Eye View, a 3D globe.",
  'Call the tool or tools needed to satisfy the user. Do not invent tool names or arguments.',
  'For ordinary conversation that is not a control request, answer normally and call no tool.',
  'Prefer one tool per intent. A request naming two intents gets two calls.',
].join('\n');

/** Realtime tool defs are flat; chat-completions nests them under `function`. */
function toChatCompletionsTools(tools) {
  return tools.map(({ type, name, description, parameters }) => ({
    type: type || 'function',
    function: { name, description, parameters },
  }));
}

/**
 * Evaluate one phrase's expectation against the tool calls the model produced.
 *
 * Mirrors the sibling harness's semantics: `expect` is a tool name, an array meaning
 * "all of these", or {oneOf:[...]}; `expectNone` pins turns that must call nothing.
 *
 * @returns {{ok: boolean, reason: string}}
 */
function judge(entry, calledNames) {
  if (entry.expectNone) {
    return calledNames.length === 0
      ? { ok: true, reason: 'no tools, as required' }
      : { ok: false, reason: `expected no tool call, got ${calledNames.join(', ')}` };
  }
  const expect = entry.expect;
  if (typeof expect === 'string') {
    return calledNames.includes(expect)
      ? { ok: true, reason: expect }
      : { ok: false, reason: `expected ${expect}, got ${calledNames.join(', ') || 'none'}` };
  }
  if (Array.isArray(expect)) {
    const missing = expect.filter((name) => !calledNames.includes(name));
    return missing.length === 0
      ? { ok: true, reason: expect.join(' + ') }
      : { ok: false, reason: `missing ${missing.join(', ')}; got ${calledNames.join(', ') || 'none'}` };
  }
  if (expect && Array.isArray(expect.oneOf)) {
    const hit = expect.oneOf.find((name) => calledNames.includes(name));
    return hit
      ? { ok: true, reason: `${hit} (oneOf)` }
      : { ok: false, reason: `expected one of ${expect.oneOf.join('|')}, got ${calledNames.join(', ') || 'none'}` };
  }
  return { ok: false, reason: 'phrase entry has no usable expectation' };
}

/**
 * Spot-check arguments against the first call to the expected tool.
 * Strings match as case-insensitive substrings; everything else must be equal.
 *
 * @returns {{ok: boolean, reason: string}}
 */
function judgeArgs(entry, calls) {
  if (!entry.args) return { ok: true, reason: '' };
  const target = typeof entry.expect === 'string' ? entry.expect : null;
  const call = calls.find((c) => !target || c.name === target);
  if (!call) return { ok: false, reason: 'no matching call to check args against' };
  for (const [key, want] of Object.entries(entry.args)) {
    const got = call.args?.[key];
    if (typeof want === 'string') {
      if (!String(got ?? '').toLowerCase().includes(want.toLowerCase())) {
        return { ok: false, reason: `args.${key}: expected ~"${want}", got ${JSON.stringify(got)}` };
      }
    } else if (got !== want) {
      return { ok: false, reason: `args.${key}: expected ${JSON.stringify(want)}, got ${JSON.stringify(got)}` };
    }
  }
  return { ok: true, reason: 'args ok' };
}

/** Send one phrase and return the parsed tool calls plus timing. */
async function routePhrase(phrase, tools, systemPrompt) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  const startedAt = Date.now();
  try {
    const response = await fetch(`${BASE_URL.replace(/\/+$/, '')}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        model: MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: phrase },
        ],
        tools,
        temperature: 0,
        max_tokens: MAX_TOKENS,
      }),
    });
    const data = await response.json().catch(() => ({}));
    const choice = data?.choices?.[0] ?? {};
    const calls = (choice.message?.tool_calls ?? []).map((call) => {
      let args = {};
      try { args = JSON.parse(call.function?.arguments || '{}'); } catch { args = {}; }
      return { name: call.function?.name, args };
    });
    return {
      calls,
      ms: Date.now() - startedAt,
      finishReason: choice.finish_reason ?? null,
      completionTokens: data?.usage?.completion_tokens ?? null,
      error: response.ok ? null : (data?.error?.message || `HTTP ${response.status}`),
    };
  } catch (error) {
    return {
      calls: [], ms: Date.now() - startedAt, finishReason: null, completionTokens: null,
      error: error?.name === 'AbortError' ? `timeout after ${TIMEOUT_MS}ms` : String(error?.message || error),
    };
  } finally {
    clearTimeout(timer);
  }
}

async function main() {
  const tools = toChatCompletionsTools(GEV_REALTIME_TOOLS);
  const systemPrompt = {
    slim: SLIM_INSTRUCTIONS,
    local: GEV_LOCAL_VOICE_INSTRUCTIONS,
    full: GEV_REALTIME_INSTRUCTIONS,
  }[PROMPT_MODE] ?? GEV_LOCAL_VOICE_INSTRUCTIONS;
  const phrases = ONLY
    ? PHRASES.filter((p) => p.phrase.toLowerCase().includes(ONLY.toLowerCase()))
    : PHRASES;

  console.log(`endpoint : ${BASE_URL}`);
  console.log(`model    : ${MODEL}`);
  console.log(`prompt   : ${PROMPT_MODE} (${systemPrompt.length} chars)`);
  console.log(`tools    : ${tools.length}`);
  console.log(`phrases  : ${phrases.length}${ONLY ? ` (filtered by "${ONLY}")` : ''}\n`);

  const rows = [];
  let pass = 0;
  for (const entry of phrases) {
    const result = await routePhrase(entry.phrase, tools, systemPrompt);
    const names = result.calls.map((c) => c.name).filter(Boolean);
    let verdict = result.error
      ? { ok: false, reason: `request failed: ${result.error}` }
      : judge(entry, names);
    if (verdict.ok) {
      const argVerdict = judgeArgs(entry, result.calls);
      if (!argVerdict.ok) verdict = argVerdict;
    }
    if (verdict.ok) pass += 1;
    const truncated = result.finishReason === 'length' ? ' [TRUNCATED]' : '';
    console.log(
      `${verdict.ok ? 'PASS' : 'FAIL'}  ${String(result.ms).padStart(6)}ms  `
      + `${entry.phrase.slice(0, 52).padEnd(52)}  ${verdict.reason}${truncated}`
    );
    rows.push({ phrase: entry.phrase, expect: entry.expect ?? null, expectNone: !!entry.expectNone,
      called: names, args: result.calls.map((c) => c.args), ok: verdict.ok, reason: verdict.reason,
      ms: result.ms, finishReason: result.finishReason, completionTokens: result.completionTokens });
  }

  const failures = rows.filter((r) => !r.ok);
  const times = rows.map((r) => r.ms).sort((a, b) => a - b);
  const median = times.length ? times[Math.floor(times.length / 2)] : 0;
  console.log(`\n${pass}/${phrases.length} passed  (${((pass / phrases.length) * 100).toFixed(1)}%)`);
  console.log(`median ${median}ms  p90 ${times[Math.floor(times.length * 0.9)] ?? 0}ms  max ${times.at(-1) ?? 0}ms`);
  if (rows.some((r) => r.finishReason === 'length')) {
    console.log('NOTE: some turns hit max_tokens. Raise --max-tokens, or disable reasoning '
      + "on the server with --chat-template-args '{\"enable_thinking\":false}'.");
  }
  if (failures.length) {
    console.log(`\nfailures (${failures.length}):`);
    for (const f of failures) console.log(`  - ${f.phrase}\n      ${f.reason}`);
  }

  if (JSONL) {
    const out = path.resolve(__dirname, '..', JSONL);
    fs.mkdirSync(path.dirname(out), { recursive: true });
    fs.writeFileSync(out, rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
    console.log(`\nevidence: ${out}`);
  }
  process.exitCode = failures.length ? 1 : 0;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
