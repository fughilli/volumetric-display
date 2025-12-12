// Keep-alive white noise burst for speaker timeout prevention
// Usage: chuck keepalive.ck

// Audio chain - white noise with smooth envelope
Noise noise => ADSR envelope => Gain volume => dac;

// Set volume to 20%
volume.gain(0.05);

// Set up the envelope - smooth onset, sustain, and offset
// Attack: 0.5s, Decay: 0ms, Sustain: 1.0 (full), Release: 0.5s
envelope.set(0.2, 0.0, 0.1, 0.4);

// Trigger the keep-alive sound
envelope.keyOn();
0.3::second => now; // Sustain for 0.5s
envelope.keyOff();
0.4::second => now; // Release time 0.5s
