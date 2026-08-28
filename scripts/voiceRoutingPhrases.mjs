/**
 * voiceRoutingPhrases.mjs — the shared voice-routing expectation table.
 *
 * Extracted from qa-voice-routing.mjs so that harness and the local-model eval
 * (qa-voice-routing-local.mjs) assert against ONE table. Importing the harness
 * itself is not an option: it has top-level side effects and launches puppeteer.
 *
 * Kept as pure data — no imports, no side effects.
 */

/**
 * Phrase table. `expect` = tool name(s) the model must call (string, or array
 * meaning "all of these", or {oneOf:[...]} for documented acceptable variance).
 * `expectNone` pins conversational turns that must NOT call tools.
 * `args` = spot-check subset matched against the first matching call's args
 * (substring match for strings, exact for booleans/numbers).
 */
export const PHRASES = [
  // — navigation & framing —
  { phrase: 'Take me to Tokyo', expect: 'fly_to_location' },
  { phrase: 'Fly to the Golden Gate Bridge', expect: 'fly_to_location' },
  { phrase: 'Go to Sixth Street in Austin', expect: 'fly_to_location' },
  { phrase: 'Show me the Alps from above', expect: { oneOf: ['fly_to_location', 'frame_overhead'] } },
  { phrase: 'Zoom in a bit', expect: 'adjust_camera_zoom' },
  { phrase: 'Zoom out a little', expect: 'adjust_camera_zoom' },
  { phrase: 'Zoom out to a globe view', expect: 'zoom_to_globe' },
  { phrase: 'Show me the whole earth', expect: 'zoom_to_globe' },
  { phrase: 'Frame the aircraft near us from overhead', expect: 'frame_overhead' },

  // — the satellites trap: data layer, never basemap —
  { phrase: 'Show me the satellites', expect: { oneOf: ['set_layer_visibility', 'frame_overhead'] } },
  { phrase: 'Turn off the satellites', expect: 'set_layer_visibility', args: { layerId: 'satellites' } },
  { phrase: 'Switch to Bing aerial', expect: 'set_map_stack' },
  { phrase: 'Switch the basemap to OSM', expect: 'set_map_stack' },

  // — layers —
  { phrase: 'Turn on the flights layer', expect: 'set_layer_visibility', args: { layerId: 'flights' } },
  { phrase: 'Show me live vessels', expect: 'set_layer_visibility' },
  { phrase: 'Turn on the fires layer', expect: 'set_layer_visibility' },
  { phrase: 'Turn on street traffic', expect: 'set_layer_visibility', args: { layerId: 'traffic' } },
  { phrase: 'Open the data layers menu', expect: 'show_data_layers_menu' },
  { phrase: 'Show me the datacenter layers', expect: 'show_data_layers_menu' },
  { phrase: 'Turn on the datacenters layer', expect: 'set_layer_visibility' },

  // — visual styles & post-fx —
  { phrase: 'Give me night vision', expect: 'set_visual_style' },
  { phrase: 'Switch to thermal view', expect: 'set_visual_style' },
  { phrase: 'Back to the normal look', expect: 'set_visual_style' },
  { phrase: 'Turn on bloom', expect: 'set_post_processing' },
  { phrase: 'Sharpen the image a touch', expect: 'set_post_processing' },

  // — HUD / detection / panels —
  { phrase: 'Turn the HUD off', expect: 'set_hud' },
  { phrase: 'Switch to the tactical layout', expect: 'set_hud' },
  { phrase: 'Turn on detection', expect: 'set_detection' },
  { phrase: 'Set detection density to fifty percent', expect: 'set_detection' },

  // — context questions —
  { phrase: 'What am I looking at right now?', expect: 'get_entity_context' },
  { phrase: 'What city is this below us?', expect: 'get_entity_context' },
  { phrase: 'Is there anything interesting in view?', expect: 'get_entity_context' },

  // — tracking —
  // Context-free text turns may reasonably look before tracking; either
  // routing is correct (production sessions always carry screen context).
  { phrase: 'Track that plane', expect: { oneOf: ['track_entity', 'get_entity_context'] } },
  { phrase: 'Follow the nearest aircraft', expect: { oneOf: ['track_entity', 'get_entity_context'] } },
  { phrase: 'Stop tracking', expect: 'stop_tracking' },

  // — CCTV / scenes / ISS —
  { phrase: 'Show me the nearest traffic camera', expect: 'control_cctv' },
  { phrase: 'Turn on the camera viewsheds', expect: 'control_cctv' },
  { phrase: 'Play a news radio station near Austin', expect: 'control_radio', args: { action: 'select', category: 'news', locationId: 'austin' } },
  { phrase: 'Turn on the radio', expect: 'control_radio', args: { action: 'play' } },
  { phrase: 'Set the radio volume to thirty percent', expect: 'control_radio', args: { action: 'volume', volumePct: 30 } },
  { phrase: 'Pause the radio', expect: 'control_radio', args: { action: 'pause' } },
  { phrase: 'Stop the radio', expect: 'control_radio', args: { action: 'stop' } },
  { phrase: 'When does the ISS pass over next?', expect: 'next_iss_pass' },

  // — annotations —
  { phrase: 'Annotate the Texas State Capitol and its grounds', expect: 'annotate_map' },
  { phrase: 'Outline the state of Texas', expect: 'annotate_map' },
  { phrase: 'Outline Lady Bird Lake', expect: 'annotate_map' },
  { phrase: 'Draw the walking route from the Capitol to Zilker Park', expect: 'annotate_map' },
  { phrase: 'How far is the Eiffel Tower from the Louvre?', expect: 'annotate_map' },
  { phrase: 'Clear the map', expect: 'clear_annotations' },

  // — multi-intent (assert ALL tools fire before speech) —
  {
    phrase: 'Switch to night vision and turn on the flights layer',
    expect: ['set_visual_style', 'set_layer_visibility'],
  },
  {
    phrase: 'Turn off the HUD and take me to Paris',
    expect: ['set_hud', 'fly_to_location'],
  },
  {
    phrase: 'Go to full planet view and then turn on the radio',
    expect: ['zoom_to_globe', 'control_radio'],
    argsByTool: { control_radio: { action: 'play' } },
  },

  // — camera verbs (tools #23/#24) —
  { phrase: 'Orbit around this area slowly', expect: 'move_camera' },
  { phrase: 'Pan left a bit', expect: 'move_camera' },
  { phrase: 'Stop moving the camera', expect: 'move_camera' },
  { phrase: 'Fly the route we just drew', expect: 'fly_route' },

  // — analyst queries (tool #22) —
  { phrase: 'How many flights are over Texas right now?', expect: 'analyst_query' },
  { phrase: 'Which ships are headed to Oakland?', expect: 'analyst_query' },
  { phrase: 'What is the biggest fire near Los Angeles?', expect: 'analyst_query' },
  { phrase: 'Is anything flying above forty thousand feet?', expect: 'analyst_query' },

  // — negative controls: conversation must NOT tool-call —
  { phrase: 'How is your evening going?', expectNone: true },
  { phrase: 'Tell me a fun fact about maps', expectNone: true },
];
