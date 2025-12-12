// Swoosh sound effect for movement through hoops
// Usage: chuck swoosh.ck

// Audio chain - create a swoosh sound
Noise noise => BPF filter => ADSR envelope => dac;

// Set up the filter - sweep through frequencies for swoosh effect
filter.set(2000, 1.5); // Bandpass for swoosh

// Set up the envelope - longer attack for swoosh effect
envelope.set(0.1, 0.3, 0.0, 0.4); // Gradual attack for swoosh

// Trigger the swoosh
envelope.keyOn();
0.4::second => now;
envelope.keyOff();
0.5::second => now;
