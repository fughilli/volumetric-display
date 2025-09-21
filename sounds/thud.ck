// Thud sound effect for impacts
// Usage: chuck thud.ck

// Audio chain
Impulse thud => ResonZ filter => ADSR envelope => dac;

// Set up the filter
filter.set(150, 0.05); // Resonant frequency and Q

// Set up the envelope
envelope.set(0.001, 0.3, 0.0, 0.4); // Attack, Decay, Sustain, Release

// Trigger the thud
thud.next(1.0);
envelope.keyOn();
0.4::second => now;
envelope.keyOff();
0.6::second => now;
