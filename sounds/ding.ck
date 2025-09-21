// Ding sound effect for scoring
// Usage: chuck ding.ck

// Audio chain
TriOsc ding => ADSR envelope => dac;

// Set up the oscillator
ding.freq(1200); // Frequency in Hz

// Set up the envelope
envelope.set(0.01, 0.2, 0.0, 0.3); // Attack, Decay, Sustain, Release

// Trigger the ding
envelope.keyOn();
0.3::second => now;
envelope.keyOff();
0.5::second => now;
