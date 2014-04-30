package NeuralNet.old;

import java.util.Random;

import static java.lang.Math.exp;

/**
 * User: Vasily
 * Date: 28.04.14
 * Time: 15:55
 */
public class Layer {
    double[][] weights;
    private int inputSize;
    private int outputSize;
    private double[] input;
    private double[] mappedInput;
    private double gamma = 1e-4; //gradient step
    private Random random = new Random();
    private double[] errors;


    public Layer(int inputSize, int outputSize) {
        weights = new double[outputSize][inputSize];
        for (int i = 0; i < outputSize; ++i)
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] = random.nextGaussian();
            }
        this.inputSize = inputSize;
        this.mappedInput = new double[outputSize];
        this.errors = new double[outputSize];
        this.outputSize = outputSize;
    }

    public double sigmoid(double value) {
        return 1.0 / (1.0 + exp(value));
    }

    public double diffSigmoid(double value) {
        double sigm = sigmoid(value);
        return sigm * (1 - sigm);
    }

    // \Sum NN(i)log(NN(i)
    public double[] forward(double[] input) {
        this.input = input;

        double[] output = new double[outputSize];
        if (input.length != inputSize) {
            System.err.println("Error");
            return input;
        }

        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                output[i] += weights[i][j] * input[j];
            }
        }
        System.arraycopy(output, 0, mappedInput, 0, outputSize);
        for (int i = 0; i < outputSize; ++i) {
            output[i] = sigmoid(output[i]);
        }
        return output;
    }

    public void backprop(Layer nextLayer) {
        errors = new double[outputSize];
        for (int i = 0; i < errors.length;++i) {
            for (int j=0;j< nextLayer.errors.length; ++j) {
                errors[i] += nextLayer.weights[j][i]*nextLayer.errors[j];
            }
            errors[i] *= mappedInput[i];
        }
        for (int i=0;i< outputSize;++i)
            for (int j=0; j < inputSize;++j) {
                weights[i][j] -= gamma*errors[i] * input[j];
            }
    }

    public void setError(double[] error) {
        this.errors = error;
    }
}
