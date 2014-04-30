package NeuralNet.old;

import java.util.ArrayList;
import java.util.Random;

import static java.lang.Math.log;

/**
 * User: Vasily
 * Date: 28.04.14
 * Time: 15:53
 */
public class NeuralNet {
    ArrayList<Layer> layers;
    private double threshold = 0.5;
    private Random random = new Random();
    private double eps = 1e-2;

    public NeuralNet(int[] networkTopology) {
        layers = new ArrayList<>(networkTopology.length);
        for (int i = 1; i < networkTopology.length; ++i) {
            layers.add(new Layer(networkTopology[i - 1], networkTopology[i]));
        }
    }

    public double predict(double[] input) {
        double[] current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        return current[0];
    }

    public void fit(double[] observation, int label) {
        double firstProb = predict(observation);
        double[] error = new double[1];
        if (label == 0) {
            error[0] = -1.0 / firstProb;
        } else {
            error[0] = 1.0 / ( 1 - firstProb);
        }
        layers.get(layers.size()-1).setError(error);
        for (int i = layers.size() - 2; i >= 0; --i) {
            layers.get(i).backprop(layers.get(i+1));
        }
    }

    public void fit(double[][] observations, int[] label) {
        double[] error = new double[1];
        for (int i =0; i< observations.length;++i) {
            double firstProb = predict(observations[i]);
            if (label[i] == 0) {
                error[0]  -= -1.0 / firstProb;
            } else {
                error[0] -= 1.0 / (1 - firstProb);
            }
        }
        layers.get(layers.size()-1).setError(error);
        for (int i = layers.size() - 2; i >= 0; --i) {
            layers.get(i).backprop(layers.get(i+1));
        }
    }

    public double fitNeuralNet(double[][] learn, double[][] validate, int[] learnAnswers, int[] validateAnswers) {
        double valError;
        double newError = 1e10;
        do {
            valError = newError;
            for (int i = 0; i < 10; ++i) {
                fit(learn,learnAnswers);
            }

            System.out.println("10 iterations done");

            newError = 0;
            for (int i = 0; i < learn.length; ++i) {
                double prob = predict(learn[i]);
                if (learnAnswers[i] == 0) {
                    newError -= log(prob > 1e-10? prob : 1e-10);
                } else {
                    newError -= log(1 - prob > 1e-10? 1 - prob : 1e-10);
                }
            }
            System.out.println(String.format("new error %f",newError)) ;
        } while (newError < valError - eps);
        return valError;
    }

    public int[] predictTest(double[][] test) {
        int[] answer = new int[test.length];
        for (int i = 0; i < test.length; ++i) {
            answer[i] = predict(test[i]) > threshold ? 0 : 1;
        }
        return answer;
    }
}

//
//valError = newError;
//        for (int i = 0; i < 1000; ++i) {
//        int observationId = random.nextInt(learn.length);
//        fit(learn[observationId], learnAnswers[observationId] == 0 ? 0 : 1);
//        }
//
//        System.out.println("1000 iterations done");
//
//        newError = 0;
//        for (int i = 0; i < validate.length; ++i) {
//        double prob = predict(validate[i]);
//        if (validateAnswers[i] == 0) {
//        newError -= log(prob > 1e-10? prob : 1e-10);
//        } else {
//        newError -= log(1 - prob > 1e-10? 1 - prob : 1e-10);
//        }
//        }
//        System.out.println(String.format("new error %f",newError))