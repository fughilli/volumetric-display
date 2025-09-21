// Sound Server for Game Audio
// Usage: chuck sound_server.ck
// Send OSC messages to trigger sounds:
// /sound/click, /sound/pop, /sound/beep, /sound/crunch, /sound/warble, /sound/woosh, /sound/ding, /sound/thud

// OSC receiver
OscIn oin;
OscMsg msg;
6449 => oin.port;
oin.listenAll();

// Sound effect functions
fun void playClick() {
    Impulse click => ResonZ filter => ADSR envelope => dac;
    filter.set(800, 0.1);
    envelope.set(0.001, 0.05, 0.0, 0.1);
    click.next(1.0);
    envelope.keyOn();
    0.1::second => now;
    envelope.keyOff();
    0.2::second => now;
}

fun void playPop() {
    Noise noise => BPF filter => ADSR envelope => dac;
    filter.set(2000, 0.5);
    envelope.set(0.01, 0.1, 0.0, 0.15);
    envelope.keyOn();
    0.15::second => now;
    envelope.keyOff();
    0.2::second => now;
}

fun void playBeep() {
    SinOsc beep => ADSR envelope => dac;
    beep.freq(800);
    envelope.set(0.01, 0.1, 0.5, 0.2);
    envelope.keyOn();
    0.3::second => now;
    envelope.keyOff();
    0.4::second => now;
}

fun void playCrunch() {
    Noise noise => LPF filter => ADSR envelope => dac;
    filter.set(500);
    envelope.set(0.02, 0.2, 0.0, 0.3);
    envelope.keyOn();
    0.3::second => now;
    envelope.keyOff();
    0.5::second => now;
}

fun void playWarble() {
    SinOsc osc => ADSR envelope => dac;
    osc.freq(400);
    envelope.set(0.05, 0.3, 0.3, 0.4);
    envelope.keyOn();
    0.8::second => now;
    envelope.keyOff();
    1.0::second => now;
}

fun void playWoosh() {
    Noise noise => HPF filter => ADSR envelope => dac;
    filter.set(1000);
    envelope.set(0.1, 0.4, 0.0, 0.5);
    envelope.keyOn();
    0.5::second => now;
    envelope.keyOff();
    0.8::second => now;
}

fun void playDing() {
    TriOsc ding => ADSR envelope => dac;
    ding.freq(1200);
    envelope.set(0.01, 0.2, 0.0, 0.3);
    envelope.keyOn();
    0.3::second => now;
    envelope.keyOff();
    0.5::second => now;
}

fun void playThud() {
    Impulse thud => ResonZ filter => ADSR envelope => dac;
    filter.set(150, 0.05);
    envelope.set(0.001, 0.3, 0.0, 0.4);
    thud.next(1.0);
    envelope.keyOn();
    0.4::second => now;
    envelope.keyOff();
    0.6::second => now;
}

// Main event loop
while (true) {
    oin => now;
    while (oin.recv(msg)) {
        msg.address => string address;

        if (address == "/sound/click") {
            spork ~ playClick();
        } else if (address == "/sound/pop") {
            spork ~ playPop();
        } else if (address == "/sound/beep") {
            spork ~ playBeep();
        } else if (address == "/sound/crunch") {
            spork ~ playCrunch();
        } else if (address == "/sound/warble") {
            spork ~ playWarble();
        } else if (address == "/sound/woosh") {
            spork ~ playWoosh();
        } else if (address == "/sound/ding") {
            spork ~ playDing();
        } else if (address == "/sound/thud") {
            spork ~ playThud();
        }
    }
}
