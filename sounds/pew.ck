// Pew sound effect for raygun/laser shooting
// Usage: chuck pew.ck

// Audio chain - create a laser/raygun sound
SawOsc laser => LPF filter => ADSR envelope => dac;

// Set up the oscillator
laser.freq(1200); // Higher frequency for laser sound

// Set up the filter for laser effect
filter.set(3000, 2.0); // Low pass with high Q for resonance

// Set up the envelope - quick attack, short duration
envelope.set(0.001, 0.05, 0.0, 0.1); // Very quick attack, no sustain

// Trigger the pew
envelope.keyOn();
0.1::second => now;
envelope.keyOff();
0.15::second => now;
