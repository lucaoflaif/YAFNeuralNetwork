#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <fstream>
#include <sstream>
#include <string>

#include <random>

std::random_device rd; // obtain a random number from hardware
std::mt19937 gen(rd()); // seed the generator
std::uniform_real_distribution<> distr(-1, 1); // define the range

const int MAX_DIM = 784;

double weights[4][MAX_DIM][MAX_DIM];
double gradientW[4][MAX_DIM][MAX_DIM];
double gradientB[4][MAX_DIM];

static int numOfTrainedRecords = 0;
static int numOfTrainedBatches = 0;

class Neuron {
    public:
    double weightedInputValue;
    double activationValue;
    double bias;
};

class Layer {
    int numOutputNeurons;

    public:
        std::vector<Neuron> neurons;
        Layer(int size, int sizeOutput) {
            numOutputNeurons = sizeOutput;

            for (int i = 0; i < size; i++) {
                neurons.push_back(Neuron());
            }
        }
        
};

class NeuralNetwork {
    public:
        std::vector<Layer> layers;
        
        NeuralNetwork(std::vector<int> sizes) {
            printf("inside");
            for (int i = 0; i < sizes.size()-1; i++) {
                layers.push_back(Layer(sizes[i], sizes[i+1]));
            }
            //add the last layer
            layers.push_back(Layer(sizes.back(), 0));
        }

        void initializeWeights() {
            //weights[layer][indx neuron of current layer][index neuron of next layer]
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < MAX_DIM; j++) {
                    for (int k = 0; k < MAX_DIM; k++) {
                        weights[i][j][k] = distr(gen);
                    }
                }
            }
        }

        void initializeBias() {
            for (auto & layer : layers) {
                for (auto& neuron : layer.neurons) {
                    neuron.bias = 0;
                }
            }
        }

        double activationFunction(double input) {
            //sigmoid
            return 1 / (1+ std::exp(-input));
        }

        double activationFunctionDerivative(double input) {
            //sigmoid derivative
            return activationFunction(input) * (1 - activationFunction(input));
        }

        void setInputLayerNodeValues(std::vector<double> inputValues) {
            Layer & inputLayer = layers[0];

            int i = 0;
            for (auto &neuron : inputLayer.neurons) {
                neuron.activationValue = inputValues[i];

                i++;
            }
        }

        void calculateActivationValues() {
            //we start from one because zero is the input layer
            for (int i = 1; i < layers.size(); i++) {
                for (int j = 0; j < layers[i].neurons.size(); j++) {
                    Layer& prevLayer = layers[i-1];

                    //loop thourgh the previous layer and sum all the weighted activation values
                    double sum = 0;
                    for (int k = 0; k < prevLayer.neurons.size(); k++) {
                        sum += prevLayer.neurons[k].activationValue * weights[i-1][k][j];
                    }

                    //then assign to the current neuron of the current layer the sum plus the bias
                    layers[i].neurons[j].weightedInputValue = sum + layers[i].neurons[j].bias;

                    //now we pass the weighted input to the activation function to calculate the activation value for each neuron
                    layers[i].neurons[j].activationValue = activationFunction(layers[i].neurons[j].weightedInputValue);
            }
        }
    }

        std::vector<double> nodeValuesOutputLayer(std::vector<double> expectedOutputs) {
            int layersSize = layers.size();
            
            std::vector<double> nodeValues;

            Layer& outputLayer = layers.back();
            Layer& prevLayer = layers[layersSize - 2];

            int i = 0;
            for (auto &neuron : outputLayer.neurons) {
                
                //derivative of the activation function - derivative of cost over the activation value [2*(a-y)]
                double derivative = activationFunctionDerivative(neuron.weightedInputValue);
                double diff = neuron.activationValue - expectedOutputs[i];
                double value = derivative * 2*diff;

                nodeValues.push_back(value);

                i ++;
            }
            return nodeValues;
        }

        std::vector<double> nodeValuesHiddenLayer(int hiddenLayerIndex, std::vector<double> nextLayerNodeValues) {
            std::vector<double> nodeValues;

            Layer currentLayer = layers[hiddenLayerIndex];
            Layer nextLayer = layers[hiddenLayerIndex + 1];

            int currentHiddenLayerNeuronIndex = 0;
            int nextLayerNeuronIndex = 0;

            double currentNodeValue = 0;

            for (auto& currentHiddenLayerNeuron : currentLayer.neurons) {
                for (auto& nextLayerNeuron : nextLayer.neurons) {
                    currentNodeValue += nextLayerNodeValues[nextLayerNeuronIndex] * weights[hiddenLayerIndex][currentHiddenLayerNeuronIndex][nextLayerNeuronIndex];

                    nextLayerNeuronIndex++;
                }
                currentNodeValue = currentNodeValue * activationFunctionDerivative(currentHiddenLayerNeuron.activationValue);
                nodeValues.push_back(currentNodeValue);
                currentNodeValue = 0;
                nextLayerNeuronIndex=0;

                currentHiddenLayerNeuronIndex++;
            }
            return nodeValues;
        }

        void calculateGradientW(int currentLayerIndex, std::vector<double> CurrentLayerNodeValues) {

            //in this case nextLayer is output layer
            Layer& prevLayer = layers[currentLayerIndex-1];
            Layer& currentLayer = layers[currentLayerIndex];

            int prevLayerNeuronIndex = 0;
            int currentLayerNeuronIndex = 0;

            for (auto& currentLayerNeuron : currentLayer.neurons) {
                for (auto& prevLayerNeuron : prevLayer.neurons) {
                    
                    gradientW[currentLayerIndex-1][prevLayerNeuronIndex][currentLayerNeuronIndex] += (CurrentLayerNodeValues[currentLayerNeuronIndex] * prevLayerNeuron.activationValue)*(0.1);

                    prevLayerNeuronIndex++;
                }
                prevLayerNeuronIndex = 0;
                currentLayerNeuronIndex++;
            }

        }

        void calculateGradientB(int currentLayerIndex, std::vector<double> nodeValues) {
            for (int i = 0; i < layers[currentLayerIndex].neurons.size(); i++) {
                gradientB[currentLayerIndex][i] += nodeValues[i]*0.1;
            }        
        }

        void updateAllGradients(){
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < MAX_DIM; j++) {
                    for (int k = 0; k < MAX_DIM; k++) {
                        weights[i][j][k] -= gradientW[i][j][k]*0.5;
                    }
                }
            }

            for (int i = 0; i < layers.size(); i++) {
                for (int k = 0; k < layers[i].neurons.size(); k++) {
                    layers[i].neurons[k].bias -= gradientB[i][k]*0.5;
                }
            }
        }

        void resetAllGradients() {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < MAX_DIM; j++) {
                    for (int k = 0; k < MAX_DIM; k++) {
                        gradientW[i][j][k] = 0;
                    }
                }
            }

            for (int i = 0; i < layers.size(); i++) {
                for (int k = 0; k < layers[i].neurons.size(); k++) {
                    gradientB[i][k] = 0;
                }
            }
        }

        int Classify(std::vector<double> inputs) {
            setInputLayerNodeValues(inputs);
            calculateActivationValues();

            std::vector<double> outputNeuronsValues;

            for (auto neuron : layers[layers.size()-1].neurons) {
                outputNeuronsValues.push_back(neuron.activationValue);
            }
            int maxValueIndex = std::distance(outputNeuronsValues.begin(),std::max_element(outputNeuronsValues.begin(), outputNeuronsValues.end()));
            return maxValueIndex;
        }
};


int trainAndBackpropagation(NeuralNetwork& nn, std::vector <double> recordData, int BATCH_SIZE) {
    //create and intiialize the expected outputs vector
        std::vector<double> expectedOutputs;

        //we calculate the expected output number of the data line wich is in the first position then we create outputs
        int outputNumberValue = recordData.front()*255;
        for (int i = 0; i < 10; i++) {
            i == outputNumberValue ? expectedOutputs.push_back(1) : expectedOutputs.push_back(0);
        }
        
        //delete first record (it's the output)
        recordData.erase(recordData.begin());

        //so now the record vector is the input vector
        //std::vector<double> inputValues = recordData;

        //feed the neural network with inputs and propagate forward
        nn.setInputLayerNodeValues(recordData);
        nn.calculateActivationValues();

        std::vector<double> nodeValuesOutput = nn.nodeValuesOutputLayer(expectedOutputs);
        nn.calculateGradientW(3, nodeValuesOutput);

        //clear expected outputs
        expectedOutputs.clear();
        //outputNumberValue == 0 ? expectedOutputs[0] = 0 : expectedOutputs[outputNumberValue] = 0;
    
        std::vector<double> nodeValuesNextLayer;

        std::copy(nodeValuesOutput.begin(), nodeValuesOutput.end(), std::back_inserter(nodeValuesNextLayer));

        nn.calculateGradientB(3,nodeValuesOutput);

        int j = nn.layers.size()-2;
        while (j > 0) {
            //backpropagation: we calculate the node values of the current layer with 
            //the node values of the next layer, then we calculate the gradient of weights and biases

            std::vector<double> nodeValuesCurrentLayer = nn.nodeValuesHiddenLayer(j, nodeValuesNextLayer);
            nodeValuesNextLayer.clear();
            nn.calculateGradientW(j, nodeValuesCurrentLayer);
            nn.calculateGradientB(j, nodeValuesCurrentLayer);

            std::copy(nodeValuesCurrentLayer.begin(), nodeValuesCurrentLayer.end(), std::back_inserter(nodeValuesNextLayer));
            j--;
            nodeValuesCurrentLayer.clear();
        }

        numOfTrainedRecords++;
        if (numOfTrainedRecords == BATCH_SIZE) {
            //if the batch size limit is met we can update our weights, biases and clear the gradients
            // for a new set of inputs
        
            nn.updateAllGradients();
            nn.resetAllGradients();
            numOfTrainedRecords = 0;

            numOfTrainedBatches++;
            printf("Trained batches: %d of %d\n", numOfTrainedBatches, (60000/10));
        }

    return 0;
}

void startTraining(NeuralNetwork nn, int BATCH_SIZE) {
    std::vector <double> recordData;

    std::ifstream infile_train( "MNIST_CSV/mnist_train.csv" );
    std::string token;

      while (infile_train)
      {
        //read the line until newline char is met
        std::string s;
        getline(infile_train, s , '\n');    

        //parse the single line creating a vector containing the training data
        std::stringstream ss(s);
        while (getline(ss, token, ',')) recordData.push_back(stod(token)/255);
        
        if (recordData.empty()) break;  

        //the record is ready to be fed into the nn
        trainAndBackpropagation(nn, recordData, BATCH_SIZE);
        recordData.clear();
      }
}

void startPredicting(NeuralNetwork nn) {
    std::vector <double> recordData;
    std::vector<double> inputValues;

    std::ifstream infile_train( "MNIST_CSV/mnist_test.csv" );
    std::string token;

    int numOfClassifiedInputs = 0;
    int numOfCorrectlyClassifiedInputs = 0;

    while (infile_train)
      {
        //read the line until newline char is met
        std::string s;
        getline(infile_train, s , '\n');    

        //parse the single line creating a vector containing the training data
        std::stringstream ss(s);
        while (getline(ss, token, ',')) recordData.push_back(stod(token)/255);

        const int outputValue = recordData.front()*255;

        for (int i = 1; i < recordData.size(); i++) {
            inputValues.push_back(recordData[i]);
        }
        
        if (recordData.empty()) break;  

        //the record is ready to be fed into the nn
        int predictedOutput = nn.Classify(inputValues);

        numOfClassifiedInputs++;
        if (predictedOutput == outputValue) numOfCorrectlyClassifiedInputs++;

        double accuracy = (double)numOfCorrectlyClassifiedInputs/numOfClassifiedInputs;

        if (numOfCorrectlyClassifiedInputs != 0) printf("Correct prediction: %f%\n", accuracy*100);

        recordData.clear();
        inputValues.clear();
      }
}

int main() {
    std::vector<int> sizes = {784, 100, 50, 10};
    const int BATCH_SIZE = 10;
    
    //initialize the neural network
    NeuralNetwork nn(sizes);    
    nn.initializeWeights();
    nn.initializeBias();

    startTraining(nn, BATCH_SIZE);
    startPredicting(nn);
}