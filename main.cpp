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
        


        void initializeWeights() {
            //weights[layer][indx neuron of current layer][index neuron of next layer]
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < MAX_DIM; j++) {
                    for (int k = 0; k < MAX_DIM; k++) {
                        weights[i][j][k] = distr(gen);
                        //printf("weight");
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
            //sigmoid
            return activationFunction(input) * (1 - activationFunction(input));
        }

        NeuralNetwork(std::vector<int> sizes) {
            printf("inside");
            for (int i = 0; i < sizes.size()-1; i++) {
                layers.push_back(Layer(sizes[i], sizes[i+1]));
            }
            //add the last layer
            layers.push_back(Layer(sizes.back(), 0));
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
            Layer& outputLayer = layers.back();
            std::vector<double> nodeValues;
            int layersSize = layers.size();
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
                    
                    gradientW[currentLayerIndex-1][prevLayerNeuronIndex][currentLayerNeuronIndex] += (CurrentLayerNodeValues[currentLayerNeuronIndex] * prevLayerNeuron.activationValue)*0.1;

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
            //printf("%f", gradientW[0][0][1]);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < MAX_DIM; j++) {
                    for (int k = 0; k < MAX_DIM; k++) {
                        weights[i][j][k] -= gradientW[i][j][k]*0.5;
                        //printf("w");
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
                    //layers[i].neurons[k].bias = 0;
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


std::vector<std::vector<double>> getDataset() {
    std::vector<std::vector<double>> dataset;
    std::ifstream infile( "MNIST_CSV/mnist_train.csv" );
    std::vector <double> record;
    std::string token;

      while (infile)
      {
        std::string s;
        getline(infile, s , '\n');    

        std::stringstream ss(s);
        while (getline(ss, token, ',')) {
          record.push_back(stod(token)/255);
        }
        //use record (it's a vector with all parsed integers)
        dataset.push_back(record);
        record.clear();
      }
    return dataset;
}

int main() {

    std::vector<int> sizes = {784, 100, 50, 10};
    std::vector<double> expectedOutputs;
    for (int i = 0; i < 10; i++) expectedOutputs.push_back(0);
    
    NeuralNetwork nn(sizes);    
    nn.initializeWeights();
    nn.initializeBias();

    std::vector<std::vector<double>> dataset = getDataset();

    int numImages = 0;
    for (auto image_data : dataset) {
        if (image_data.empty()) break;    

        //create outputs
        int outputNumberValue = image_data.front()*255;
        outputNumberValue == 0 ? expectedOutputs[0] = 1 : expectedOutputs[outputNumberValue] = 1;
        
        //delete first record (it's the output)
        image_data.erase(image_data.begin());

        //so now the record vector is the input vector
        std::vector<double> inputValues = image_data;

        nn.setInputLayerNodeValues(inputValues);
        nn.calculateActivationValues();

        std::vector<double> nodeValuesOutput = nn.nodeValuesOutputLayer(expectedOutputs);
        nn.calculateGradientW(3, nodeValuesOutput);
        //clear expected outputs
        outputNumberValue == 0 ? expectedOutputs[0] = 0 : expectedOutputs[outputNumberValue] = 0;
    
        std::vector<double> nodeValuesNextLayer;

        std::copy(nodeValuesOutput.begin(), nodeValuesOutput.end(), std::back_inserter(nodeValuesNextLayer));

        nn.calculateGradientB(3,nodeValuesOutput);

        int j = sizes.size()-2;
        while (j > 0) {
            std::vector<double> nodeValuesCurrentLayer = nn.nodeValuesHiddenLayer(j, nodeValuesNextLayer);
            nodeValuesNextLayer.clear();
            nn.calculateGradientW(j, nodeValuesCurrentLayer);
            nn.calculateGradientB(j, nodeValuesCurrentLayer);

            std::copy(nodeValuesCurrentLayer.begin(), nodeValuesCurrentLayer.end(), std::back_inserter(nodeValuesNextLayer));
            //std::vector<double> nodeValuesNextLayer = nodeValuesCurrentLayer;
            j--;
            nodeValuesCurrentLayer.clear();
        }

        numImages++;
        if (numImages == 10) {
            nn.updateAllGradients();
            nn.resetAllGradients();

            int predict = nn.Classify(inputValues);
            printf("input: %d, predicted %d\n", outputNumberValue, predict);
            //if (outputNumberValue == predict) printf("ok");

            numImages = 0;

        }
    }

    return 0;
}

/*
int main () {

    std::vector<int> sizes = {2, 2, 1};

    weights[0][0][0] = 0.3;
    weights[0][0][1] = -0.4;
    weights[0][1][0] = 0.2;
    weights[0][1][1] = -0.5;
    weights[1][0][0] = 0.1;
    weights[1][1][0] = 0.2;

    NeuralNetwork nn(sizes);

    std::vector<std::vector<double>> dataset {{0,0,0},{1,0,1},{1,1,0},{0,1,1}};

    for (auto& data : dataset) {

        double output = data.front();
        data.erase(data.begin());

        std::vector<double> nodeValuesNextLayer;
        nn.setInputLayerNodeValues(data);
        nn.calculateActivationValues();
        std::vector<double> expectedOutputs = {output};
        std::vector<double> nodeValuesOutput = nn.nodeValuesOutputLayer(expectedOutputs);
        

        nn.calculateGradientB(2, nodeValuesOutput);
        nn.calculateGradientW(2, nodeValuesOutput);

        nn.updateAllGradients();
        nn.resetAllGradients();

        nodeValuesNextLayer = nodeValuesOutput;

        int j = sizes.size()-2;
        while (j > 0) {
            std::vector<double> nodeValuesCurrentLayer = nn.nodeValuesHiddenLayer(j, nodeValuesNextLayer);
            nn.calculateGradientW(j, nodeValuesCurrentLayer);
            nn.calculateGradientB(j, nodeValuesCurrentLayer);
            nn.updateAllGradients();
            nn.resetAllGradients();
            nodeValuesNextLayer = nodeValuesCurrentLayer;
            j--;
        }
    }

    printf("%d \n", nn.Classify( {0,1}));
    printf("%d \n", nn.Classify( {1,1}));
    printf("%d \n", nn.Classify( {1,0}));

    return 0;
}*/