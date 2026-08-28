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
  assert.deepEqual(parsed.calls[0], { name: 'fly_to_location', args: { locationId: 'tokyo' } });
  assert.equal(parsed.calls[1].args.enabled, true);
  assert.equal(parsed.truncated, false);
});

test('malformed arguments degrade to an empty object rather than throwing', () => {
  const parsed = parseChatToolCalls({
    choices: [{ message: { tool_calls: [{ function: { name: 'zoom_to_globe', arguments: '{not json' } }] } }],
  });
  assert.deepEqual(parsed.calls, [{ name: 'zoom_to_globe', args: {} }]);
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
