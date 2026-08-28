import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  CAPTURE_SAMPLE_RATE,
  describeMicError,
  describeOutcomes,
  encodeWav,
  frameLoudness,
  parseChatToolCalls,
  readLocalVoiceStatus,
  speechThreshold,
  trimHistory,
  MAX_HISTORY_CHARS,
  SPEECH_FLOOR_MIN,
  SPEECH_FLOOR_MULTIPLIER,
} from './gevLocalVoice.js';

test('loudness separates speech-level audio from room tone', () => {
  const silence = new Float32Array(128);
  const speech = Float32Array.from({ length: 128 }, (_, i) => Math.sin(i) * 0.4);
  assert.equal(frameLoudness(silence), 0);
  assert.ok(frameLoudness(speech) > 0.2);
  assert.equal(frameLoudness(new Float32Array(0)), 0, 'an empty frame is not a divide by zero');
  assert.equal(frameLoudness(null), 0);
});

test('encodeWav writes a valid 16-bit mono PCM header', async () => {
  const samples = Float32Array.from([0, 0.5, -0.5, 1, -1]);
  const bytes = new Uint8Array(await encodeWav(samples, CAPTURE_SAMPLE_RATE).arrayBuffer());
  const view = new DataView(bytes.buffer);
  const text = (offset, length) => String.fromCharCode(...bytes.slice(offset, offset + length));

  assert.equal(text(0, 4), 'RIFF');
  assert.equal(text(8, 4), 'WAVE');
  assert.equal(text(12, 4), 'fmt ');
  assert.equal(text(36, 4), 'data');
  assert.equal(view.getUint16(20, true), 1, 'PCM format tag');
  assert.equal(view.getUint16(22, true), 1, 'mono');
  assert.equal(view.getUint32(24, true), CAPTURE_SAMPLE_RATE);
  assert.equal(view.getUint16(34, true), 16, 'bits per sample');
  assert.equal(view.getUint32(40, true), samples.length * 2, 'data chunk size');
  assert.equal(bytes.length, 44 + samples.length * 2);
});

test('encodeWav clamps out-of-range samples instead of wrapping', async () => {
  const bytes = new Uint8Array(await encodeWav(Float32Array.from([2, -2])).arrayBuffer());
  const view = new DataView(bytes.buffer);
  assert.equal(view.getInt16(44, true), 32767, 'positive overshoot clamps to full scale');
  assert.equal(view.getInt16(46, true), -32768, 'negative overshoot clamps, never wraps positive');
});

test('tool calls and their arguments are parsed out of a chat response', () => {
  const parsed = parseChatToolCalls({
    choices: [{
      finish_reason: 'tool_calls',
      message: {
        tool_calls: [
          { function: { name: 'fly_to_location', arguments: '{"locationId":"tokyo"}' } },
          { function: { name: 'set_layer_visibility', arguments: '{"layerId":"flights","enabled":true}' } },
        ],
      },
    }],
  });
  assert.equal(parsed.calls.length, 2);
  assert.equal(parsed.calls[0].name, 'fly_to_location');
  assert.deepEqual(parsed.calls[0].args, { locationId: 'tokyo' });
  assert.ok(parsed.calls[0].id, 'a call id is required to answer with role:"tool"');
  assert.equal(parsed.calls[1].args.enabled, true);
  assert.equal(parsed.truncated, false);
});

test('malformed arguments degrade to an empty object rather than throwing', () => {
  const parsed = parseChatToolCalls({
    choices: [{ message: { tool_calls: [{ function: { name: 'zoom_to_globe', arguments: '{not json' } }] } }],
  });
  assert.equal(parsed.calls.length, 1);
  assert.equal(parsed.calls[0].name, 'zoom_to_globe');
  assert.deepEqual(parsed.calls[0].args, {});
});

test('a truncated turn is flagged so it can be reported instead of silently dropped', () => {
  const parsed = parseChatToolCalls({
    choices: [{ finish_reason: 'length', message: { content: '' } }],
  });
  assert.equal(parsed.calls.length, 0);
  assert.equal(parsed.truncated, true);
});

test('a conversational reply carries text and no calls', () => {
  const parsed = parseChatToolCalls({ choices: [{ message: { content: 'Evening is going well.' } }] });
  assert.deepEqual(parsed.calls, []);
  assert.equal(parsed.text, 'Evening is going well.');
});

test('confirmations report what actually happened, not what was requested', () => {
  assert.equal(
    describeOutcomes([{ name: 'fly_to_location', result: { ok: true }, error: null }]),
    'Fly to location',
  );
  assert.equal(
    describeOutcomes([{ name: 'set_layer_visibility', result: null, error: 'boom' }]),
    'Set layer visibility failed',
    'a thrown tool is spoken as a failure',
  );
  assert.equal(
    describeOutcomes([{ name: 'control_radio', result: { ok: false, error: 'Radio layer unavailable' }, error: null }]),
    'Radio layer unavailable',
    'ok:false uses the tool\'s own reason rather than claiming success',
  );
  assert.equal(describeOutcomes([]), '');
});

test('multiple outcomes are spoken as one line', () => {
  assert.equal(
    describeOutcomes([
      { name: 'set_visual_style', result: { ok: true }, error: null },
      { name: 'set_layer_visibility', result: { ok: true }, error: null },
    ]),
    'Set visual style, set layer visibility',
  );
});

test('voice status reports disabled when unconfigured, unreachable, or malformed', async () => {
  const ok = await readLocalVoiceStatus(async () => ({
    ok: true, json: async () => ({ enabled: true, model: 'qwen' }),
  }));
  assert.deepEqual(ok, { enabled: true, model: 'qwen' });

  const off = await readLocalVoiceStatus(async () => ({ ok: true, json: async () => ({ enabled: false }) }));
  assert.deepEqual(off, { enabled: false, model: null });

  const http503 = await readLocalVoiceStatus(async () => ({ ok: false, status: 503 }));
  assert.equal(http503.enabled, false);

  const threw = await readLocalVoiceStatus(async () => { throw new Error('offline'); });
  assert.equal(threw.enabled, false, 'a dead server must not break voice init');
});

test('microphone failures name a cause the user can act on', () => {
  const denied = describeMicError({ name: 'NotAllowedError' });
  assert.match(denied, /address bar/, 'points at the site permission');
  assert.match(denied, /System Settings/, 'and at the OS grant, which is the other common cause');

  assert.match(describeMicError({ name: 'NotFoundError' }), /No microphone/);
  assert.match(describeMicError({ name: 'NotReadableError' }), /in use by another app/);
  assert.match(describeMicError({ name: 'AbortError' }), /Try again/);
  assert.match(describeMicError({ name: 'WeirdNewError' }), /WeirdNewError/, 'unknown names are surfaced, not swallowed');
  assert.match(describeMicError(undefined), /unknown error/);
});

test('the speech gate adapts to the room but never drops below the floor', () => {
  // Silent room: the floor keeps the gate off the noise.
  assert.equal(speechThreshold(0), SPEECH_FLOOR_MIN);
  assert.equal(speechThreshold(0.0001), SPEECH_FLOOR_MIN);

  // Noisy room: the gate rises above the measured tone rather than latching open.
  const noisy = 0.02;
  assert.equal(speechThreshold(noisy), noisy * SPEECH_FLOOR_MULTIPLIER);
  assert.ok(speechThreshold(noisy) > noisy, 'gate must sit above the noise it measured');

  // Monotonic: a louder room never yields a lower gate.
  assert.ok(speechThreshold(0.05) > speechThreshold(0.01));
});

test('a quiet mic still opens the gate at conversational level', () => {
  // A heavily noise-suppressed laptop mic idles around 0.001 and speaks near 0.02.
  const gate = speechThreshold(0.001);
  assert.ok(gate < 0.02, `speech at 0.02 must clear the gate, got ${gate}`);
});

test('missing tool_call ids are synthesized so results can still be matched back', () => {
  const parsed = parseChatToolCalls({
    choices: [{ message: { tool_calls: [
      { function: { name: 'fly_to_location', arguments: '{}' } },
      { function: { name: 'set_hud', arguments: '{}' } },
    ] } }],
  });
  assert.equal(parsed.calls.length, 2);
  assert.notEqual(parsed.calls[0].id, parsed.calls[1].id, 'synthesized ids must be distinct');
});

test('history under budget is left alone', () => {
  const history = [
    { role: 'user', content: 'fly to boston' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'a' }] },
    { role: 'tool', tool_call_id: 'a', content: '{"ok":true}' },
  ];
  assert.equal(trimHistory([...history]).length, 3);
});

test('trimming never orphans a tool message from its call', () => {
  const exchange = (n) => ([
    { role: 'user', content: `turn ${n} `.padEnd(400, 'x') },
    { role: 'assistant', content: '', tool_calls: [{ id: `c${n}` }] },
    { role: 'tool', tool_call_id: `c${n}`, content: '{"ok":true}' },
  ]);
  const history = [...exchange(1), ...exchange(2), ...exchange(3), ...exchange(4)];
  const trimmed = trimHistory(history, 900);

  assert.ok(trimmed.length < 12, 'something was dropped');
  assert.equal(trimmed[0].role, 'user', 'head must be a user turn, never a dangling tool result');
  for (let i = 0; i < trimmed.length; i += 1) {
    if (trimmed[i].role === 'tool') {
      const prior = trimmed.slice(0, i).some((m) => m.role === 'assistant');
      assert.ok(prior, 'every retained tool message still has its assistant turn ahead of it');
    }
  }
});

test('a single oversized exchange is not trimmed into nothing', () => {
  const history = [{ role: 'user', content: 'x'.repeat(MAX_HISTORY_CHARS * 2) }];
  assert.equal(trimHistory(history).length, 1, 'the current turn always survives');
});
