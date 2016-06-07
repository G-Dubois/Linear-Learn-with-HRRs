#include "State.h"

// Default constructor
State::State(){

    HRREngine hrrEngine(1024);

    reward = 0;
    hrr = hrrEngine.generateHRR();
    index = 0;
}

// Initializing constructor
State::State(double r, int v, int i) {

    HRREngine hrrEngine(v);

    reward = r;
    hrr = hrrEngine.generateHRR();
    index = i;
}


// Mutator Methods

// Get reward
double State::getReward() { return reward; }

// Get value
HRR State::getHRR() { return hrr; }

// Get index location
int State::isAt() { return index; }


// Accessor Methods

// Set reward
void State::setReward(double r) { reward = r; }

// Set index location
void State::isAt(int i) { index = i; }
