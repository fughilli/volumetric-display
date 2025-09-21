// Pop sound effect for game events
// Usage: chuck pop.ck

// Audio chain
Noise noise => BPF filter => ADSR envelope => dac;

// Set up the filter
filter.set(2000, 0.5); // Bandpass frequency and Q

// Set up the envelope
envelope.set(0.01, 0.1, 0.0, 0.15); // Attack, Decay, Sustain, Release

// Trigger the pop
envelope.keyOn();
0.15::second => now;
envelope.keyOff();
0.2::second => now;
