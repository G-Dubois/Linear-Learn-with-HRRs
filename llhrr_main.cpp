/**
 *  Author:         Grayson M. Dubois
 *  Project:        Linear Learn
 *  Description:    Agent uses TD-learning algorithm to learn the values of
 *                      states in a single array to find the goal
 */

// Dependencies

// IO Dependencies
#include <iostream>
#include <iomanip>
#include <fstream>

// Container Dependencies
#include <vector>

// Utilities
#include <random>     // For random nuber generation
#include <time.h>     // For initializing random seed
#include <limits>      // For int max
#include <cmath>

// Custom Class Dependencies
#include "State.h"
#include "hrrengine.h"

// Namespace Declarations
using namespace std;

const int INT_MAX = numeric_limits<int>::max();

// Enum Move indicates a movement left or right
enum Move {
    Left,
    Right
};

// World variables
vector<State> world;
int worldSize;
double alpha;
double lambda;
double epsilon;
double discount;
int numberOfRuns;

vector<double> eligibility(1024);
vector<double> weights(1024);

int goalLocation;
int agentLocation;

HRREngine hrrEngine(1024);

// Statistical variables
int averageNumberOfSteps;
int maxNumberOfSteps;
int minNumberOfSteps;

// Random number generator
default_random_engine randomNumberGenerator;
uniform_real_distribution<double> valueDistribution(0.0, 0.001);
uniform_real_distribution<double> epsilonSoftDistribution(0.0, 1.0);

// Function Declarations
void getSettingsFromFile(string);
Move chooseMovement(State, State, State);
Move randomMovement();
unsigned int getLeftLocation(int);
unsigned int getRightLocation(int);

// TD-learning Functions
void updateEligibility(HRR);
double V(State);
double r(State);

int main (int argc, char** argv) {

    // Set up random number generation
    randomNumberGenerator.seed( time(0) );

    // Get the world settings
    if ( argc > 1 ) {
        getSettingsFromFile(argv[1]);
    } else {
        worldSize = 64;
        alpha = 0.5;
        discount = 0.9;
    }

    // Set up a log file
    ofstream Log;
    Log.open("results.log");

    // Set up a final report file
    ofstream FinalReport;
    FinalReport.open("final.log");

    // Initialize the world array with states
    for (int i = 0; i < worldSize; i++) {

        // Create a new state and add it to the world array
        State newState(0,  i);
        world.push_back(newState);
    }

    // Initialize weight vector
    for (double& weight : weights) {
        weight = 0.0;
    }

    // Set up statistical variables
    averageNumberOfSteps = 0;
    maxNumberOfSteps = 0;
    minNumberOfSteps = INT_MAX;

    // Set up a location for the goal
    goalLocation = randomNumberGenerator() % worldSize;
    world[goalLocation].setReward(1.0);

    // Main loop of program. Run episodes until task is learned
    for (int i = 0; i < numberOfRuns; i++) {

        // Set up a location for the agent
        agentLocation = randomNumberGenerator() % worldSize;

        // Initialize Episode statistical variables
        int numberOfSteps = 0;

        // Set up a variable for the previous state
        State* thisState = &world[agentLocation];

        // Reset the eligibility vector
        fill(eligibility.begin(), eligibility.end(), 0);

        // Movement through an episode
        do {

            // Update the eligibility of the current state
            updateEligibility(thisState->getHRR());

            // If we are at the goal, calculate the td error differently, since
            //  there are no future steps
            if (thisState->isAt() == goalLocation) {
                cout << "At the goal!";
                break;
            }

            // Choose a movement for the agent
            Move movement = chooseMovement( world[getLeftLocation( agentLocation )],
                                            world[agentLocation],
                                            world[getRightLocation( agentLocation )] );

            // Perform the movement
            switch (movement) {
                case Left:
                    agentLocation = getLeftLocation(agentLocation);
                    break;
                case Right:
                    agentLocation = getRightLocation(agentLocation);
                    break;
                default:
                    agentLocation = getLeftLocation(agentLocation);
                    break;
            }

            State* nextState = &world[agentLocation];

        } while (agentLocation != goalLocation);
    }

    return 0;
}

// Get settings from config file
void getSettingsFromFile(string filename) {

    // Set up input file stream
    ifstream fin;
    fin.open(filename);

    // Do a priming read to ignore config file header
    string _;
    getline(fin, _);

    // Get the data from the input file
    fin >> worldSize >> alpha >> discount >> numberOfRuns;
}

void updateEligibility(HRR a) {
    for (int i = 0; i < eligibility.size(); i++) {
        eligibility[i] = ( a[i] + lambda * eligibility[i] ) / sqrt(2);
    }
}


// Decide what movement to make
// MOVEMENT POLICY:
//      If the values of two states are equal, choose left
//      % epsilon of the time, choose randomly (epsilon soft policy)
//      Agent MUST move
Move chooseMovement(State leftState, State currentState, State rightState) {

    // Initialize the movement to left
    Move movement = Left;

    // If the right value is greater than or equal to
    if (V(leftState) < V(rightState) ) {
        movement = Right;
    }

    // Set up epsilon soft policy
    double chanceOfRandomMovement = epsilonSoftDistribution(randomNumberGenerator);

    if (chanceOfRandomMovement < epsilon) {
        //cout << "RANDOM MOVEMENT!\n";
        movement = randomMovement();
    }

    return movement;
}

// Choose a random movement for the agent
Move randomMovement(){
    int random = randomNumberGenerator() % 2;
    switch (random){
        case 0:
            return Left;
        case 1:
            return Right;
        default:
            return Left;
    }
}

// Functions to get the left and right locations of a state

// Get left location
unsigned int getLeftLocation(int currentLocation) {
    unsigned int left;

    // If at the left bound, set left to the rightmost index
    if ( currentLocation == 0 ) {
        left = worldSize - 1;
    } else {
        left = currentLocation - 1;
    }

    return left;
}

// Get right location
unsigned int getRightLocation(int currentLocation) {
    unsigned int right;

    // If at the right bound, set right to the leftmost index
    if ( currentLocation == worldSize - 1 ) {
        right = 0;
    } else {
        right = currentLocation + 1;
    }

    return right;
}

//****** TD-Learning Functions

// Get the value of a state
double V(State s) {
    return hrrEngine.dot(s.getHRR, weights);
}

// Get the reward of a state
double r(State s) {
    return s.getReward();
}
