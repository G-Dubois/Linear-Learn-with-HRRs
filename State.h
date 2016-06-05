#ifndef STATE
#define STATE

#include <iostream>
#include <random>
#include "hrrengine.h"

using namespace std;

class State {
private:

    // Data members
    double reward;
    HRR hrr;
    int index;

public:

    // Constructors
    State();                                // Creates a generic state
    State(double r, int i);       // Initializes a state with given values

    // Accessor Methods
    double getReward();
    HRR getHRR();
    int isAt();

    // Mutator Methods
    void setReward(double r);
    void isAt(int i);
};

#endif
