// Woosh sound effect for movement
// Usage: chuck woosh.ck

// Audio chain
Noise noise => HPF filter => ADSR envelope => dac;

// Set up the filter
filter.set(1000); // High pass frequency

// Set up the envelope
envelope.set(0.1, 0.4, 0.0, 0.5); // Attack, Decay, Sustain, Release

// Trigger the woosh
envelope.keyOn();
0.5::second => now;
envelope.keyOff();
0.8::second => now;
