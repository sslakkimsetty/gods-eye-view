import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildHudSummaryBody,
  extractOpenAiResponseText,
  resolveHudSummaryTarget,
  stripReasoningBlock,
  toFiveWordHudSummary,
} from '../vite.config.js';

/** Run `fn` with a patched env, restoring the previous values afterwards. */
function withEnv(patch, fn) {
  const previous = new Map();
  for (const [key, value] of Object.entries(patch)) {
    previous.set(key, process.env[key]);
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  try {
    return fn();
  } finally {
    for (const [key, value] of previous) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

const NO_LOCAL = {
  GEV_HUD_SUMMARY_BASE_URL: undefined,
  GEV_HUD_SUMMARY_MODEL: undefined,
  GEV_HUD_SUMMARY_API_KEY: undefined,
};

test('defaults to the OpenAI Responses API when no local base URL is set', () => {
  withEnv({ ...NO_LOCAL, OPENAI_API_KEY: 'sk-test', OPENAI_HUD_SUMMARY_MODEL: 'gpt-5-nano' }, () => {
    const target = resolveHudSummaryTarget();
    assert.equal(target.local, false);
    assert.equal(target.url, 'https://api.openai.com/v1/responses');
    assert.equal(target.model, 'gpt-5-nano');
    assert.equal(target.apiKey, 'sk-test');
  });
});

test('routes to a local chat-completions server when the base URL is set', () => {
  withEnv({
    ...NO_LOCAL,
    GEV_HUD_SUMMARY_BASE_URL: 'http://127.0.0.1:11432/v1',
    GEV_HUD_SUMMARY_MODEL: 'gemma4:e4b',
  }, () => {
    const target = resolveHudSummaryTarget();
    assert.equal(target.local, true);
    assert.equal(target.url, 'http://127.0.0.1:11432/v1/chat/completions');
    assert.equal(target.model, 'gemma4:e4b');
    assert.equal(target.apiKey, null);
  });
});

test('a local base URL needs no OpenAI key, and trailing slashes do not double up', () => {
  withEnv({
    ...NO_LOCAL,
    GEV_HUD_SUMMARY_BASE_URL: 'http://127.0.0.1:11432/v1///',
    OPENAI_API_KEY: undefined,
  }, () => {
    const target = resolveHudSummaryTarget();
    assert.equal(target.url, 'http://127.0.0.1:11432/v1/chat/completions');
    assert.equal(target.apiKey, null);
  });
});

test('an explicit local API key is forwarded for gateways that require one', () => {
  withEnv({
    ...NO_LOCAL,
    GEV_HUD_SUMMARY_BASE_URL: 'http://gateway.internal/v1',
    GEV_HUD_SUMMARY_API_KEY: 'local-secret',
  }, () => {
    assert.equal(resolveHudSummaryTarget().apiKey, 'local-secret');
  });
});

test('each backend gets the request shape it expects', () => {
  const context = { place: 'Austin', layers: ['flights'] };

  const remote = buildHudSummaryBody({ model: 'gpt-5-nano', local: false }, context);
  assert.equal(remote.max_output_tokens, 100);
  assert.equal(remote.input, JSON.stringify(context));
  assert.ok(remote.instructions.includes('exactly five words'));
  assert.equal(remote.messages, undefined, 'Responses API must not receive messages');

  const local = buildHudSummaryBody({ model: 'gemma4:e4b', local: true }, context);
  assert.equal(local.max_tokens, 100);
  assert.equal(local.temperature, 0);
  assert.equal(local.messages[0].role, 'system');
  assert.ok(local.messages[0].content.includes('exactly five words'));
  assert.equal(local.messages[1].content, JSON.stringify(context));
  assert.equal(local.input, undefined, 'chat-completions must not receive input');
  assert.equal(local.max_output_tokens, undefined);
});

test('both backends are given identical instructions', () => {
  const remote = buildHudSummaryBody({ model: 'm', local: false }, {});
  const local = buildHudSummaryBody({ model: 'm', local: true }, {});
  assert.equal(local.messages[0].content, remote.instructions);
});

test('extracts text from a chat-completions response', () => {
  const text = extractOpenAiResponseText({
    choices: [{ message: { content: 'Austin downtown with live flights' } }],
  });
  assert.equal(text, 'Austin downtown with live flights');
});

test('still extracts text from both Responses API shapes', () => {
  assert.equal(extractOpenAiResponseText({ output_text: '  Tokyo harbour at dusk  ' }), 'Tokyo harbour at dusk');
  assert.equal(
    extractOpenAiResponseText({ output: [{ content: [{ text: 'London rooftops' }] }] }),
    'London rooftops',
  );
});

test('an inline reasoning block is dropped before the summary is clamped', () => {
  const raw = '<think>The user is over Austin, so I should mention it.</think>Austin skyline with live flights';
  assert.equal(stripReasoningBlock(raw), 'Austin skyline with live flights');
  assert.equal(
    toFiveWordHudSummary(extractOpenAiResponseText({ choices: [{ message: { content: raw } }] })),
    'Austin skyline with live flights',
  );
});

test('a malformed or empty local response yields no summary rather than throwing', () => {
  assert.equal(extractOpenAiResponseText({ choices: [] }), '');
  assert.equal(extractOpenAiResponseText({ choices: [{ message: { content: '   ' } }] }), '');
  assert.equal(extractOpenAiResponseText({}), '');
  assert.equal(toFiveWordHudSummary(extractOpenAiResponseText({})), '');
});
