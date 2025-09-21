// Crunch sound effect for collision events
// Usage: chuck crunch.ck

// Audio chain
Noise noise => LPF filter => ADSR envelope => dac;

// Set up the filter
filter.set(500); // Low pass frequency

// Set up the envelope
envelope.set(0.02, 0.2, 0.0, 0.3); // Attack, Decay, Sustain, Release

// Trigger the crunch
envelope.keyOn();
0.3::second => now;
envelope.keyOff();
0.5::second => now;
