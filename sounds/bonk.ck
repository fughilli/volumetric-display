// Bonk sound effect for ball impacts
// Usage: chuck bonk.ck

// Audio chain - create a bonk/thud sound
Impulse bonk => ResonZ filter => ADSR envelope => dac;

// Set up the filter for bonk sound
filter.set(400, 0.3); // Resonant frequency and Q for bonk

// Set up the envelope
envelope.set(0.001, 0.15, 0.0, 0.2); // Quick attack, short decay

// Trigger the bonk
bonk.next(1.0);
envelope.keyOn();
0.2::second => now;
envelope.keyOff();
0.3::second => now;
