// Click sound effect for UI interactions
// Usage: chuck click.ck

// Audio chain
Impulse click => ResonZ filter => ADSR envelope => dac;

// Set up the filter
filter.set(800, 0.1); // Resonant frequency and Q

// Set up the envelope
envelope.set(0.001, 0.05, 0.0, 0.1); // Attack, Decay, Sustain, Release

// Trigger the click
click.next(1.0);
envelope.keyOn();
0.1::second => now;
envelope.keyOff();
0.2::second => now;
