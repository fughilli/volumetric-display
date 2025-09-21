// Beep sound effect for notifications
// Usage: chuck beep.ck

// Audio chain
SinOsc beep => ADSR envelope => dac;

// Set up the oscillator
beep.freq(800); // Frequency in Hz

// Set up the envelope
envelope.set(0.01, 0.1, 0.5, 0.2); // Attack, Decay, Sustain, Release

// Trigger the beep
envelope.keyOn();
0.3::second => now;
envelope.keyOff();
0.4::second => now;
