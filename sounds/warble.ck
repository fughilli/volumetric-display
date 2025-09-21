// Warble sound effect for special events
// Usage: chuck warble.ck

// Audio chain
SinOsc osc => ADSR envelope => dac;

// Set up the oscillator
osc.freq(400); // Base frequency

// Set up the envelope
envelope.set(0.05, 0.3, 0.3, 0.4); // Attack, Decay, Sustain, Release

// Create warble effect with frequency modulation
SinOsc modulator => blackhole;
modulator.freq(8); // Modulation frequency
modulator.gain(50); // Modulation depth

// Trigger the warble
envelope.keyOn();
0.8::second => now;
envelope.keyOff();
1.0::second => now;
