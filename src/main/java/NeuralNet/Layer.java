package NeuralNet;

import java.util.Random;

import static java.lang.Math.exp;

/**
 * User: Vasily
 * Date: 29.04.14
 * Time: 20:47
 */
public class Layer {
    public double[] x;
    private int inputSize;
    private int outputSize;
    public double[] a;
    public double[][] weights;
    public double[][] backup;
    public double[] delta;
    private Random random = new Random();
    private double gamma = 1e-2; //step size
    private int k = 100;


    public Layer(int inputSize, int outputSize) {
        weights = new double[outputSize][inputSize];
        backup = new double[outputSize][inputSize];
        for (int i = 0; i < outputSize; ++i)
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] = 1.4*random.nextDouble() - 0.7;//random.nextGaussian();
            }
        backupWeights();
        delta = new double[outputSize];
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    public void backupWeights() {
        for (int i =0;i<weights.length;++i) {
            System.arraycopy(weights[i],0,backup[i],0,weights[i].length);
        }
    }

    public void restoreWeights() {
        for (int i =0;i<backup.length;++i) {
            System.arraycopy(backup[i],0,weights[i],0,backup[i].length);
        }
    }

    public double sigmoid(double value) {
        return 1.0 / (1.0 + exp(value));
    }

    public double diffSigmoid(double value) {
        double sigm = sigmoid(value);
        return sigm * (1 - sigm);
    }

    public double[] forward(double[] input) {
        this.x = input;
        a = new double[outputSize];
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                a[i] += weights[i][j] * x[j];
            }
        }
        double[] output = new double[outputSize];
        System.arraycopy(a, 0, output, 0, outputSize);
        for (int i = 0; i < outputSize; ++i) {
            output[i] = sigmoid(output[i]);
        }
        return output;
    }

    public void backward(Layer next) {
        decreaseGamma();
        delta = new double[outputSize];
        for (int i = 0; i < outputSize; ++i) {
            for (int s = 0; s < next.outputSize; ++s) {
                delta[i] += next.delta[s] * next.weights[s][i];
            }
            delta[i] *= diffSigmoid(a[i]);
        }

        for (int i = 0; i < outputSize; ++i)
            for (int j = 0; j < inputSize; ++j)
                weights[i][j] -= gamma * delta[i] * x[j];
    }

    public void setDelta(double[] delta) {
        this.delta = delta;
        for (int i = 0; i < outputSize; ++i)
            for (int j = 0; j < inputSize; ++j)
                weights[i][j] -= gamma * delta[i] * x[j];
    }

    public void decreaseGamma() {
        ++k;
        gamma = 1.0 / k;

    }
}
