package NeuralNet;

import java.util.ArrayList;
import java.util.Random;

import static java.lang.Math.log;

/**
 * User: Vasily
 * Date: 29.04.14
 * Time: 21:36
 */
public class NeuralNet {
    ArrayList<Layer> layers;
    private double threshold = 0.5;
    private Random random = new Random();
    private double eps = 1e-3;
    public double testKappa = -1;

    private double[][] learn;
    private int[] labels;
//
    private double[][] validate;
    private int[] validateLabels;



    public NeuralNet(int[] networkTopology, double[][] data, int[] dataLabels, double split) {
        layers = new ArrayList<>(networkTopology.length);
        for (int i = 1; i < networkTopology.length; ++i) {
            layers.add(new Layer(networkTopology[i - 1], networkTopology[i]));
        }
        createValidation(data,dataLabels,split);

    }

    public double predict(double[] input) {
        double[] current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        return current[0];
    }

    public void stochasticStep(double[] observation, int label) {
        double[] error = new double[1];
        double[] x = observation;
        for (Layer layer : layers) {
            x = layer.forward(x);
        }
        if (label == 0) {
            error[0] = -log(x[0] > 1e-10 ? x[0] : 1e-10);
        } else {
            error[0] = log(1 - x[0] > 1e-10 ? 1 - x[0] : 1e-10);
        }
        layers.get(layers.size() - 1).setDelta(error);

        for (int i = layers.size() - 2; i >= 0; --i) {
            layers.get(i).backward(layers.get(i + 1));
        }
    }



    private void createValidation(double[][] data, int[] dataLabels, double split) {

        int splitPoint  = (int)(split*dataLabels.length);
        this.learn = new double[splitPoint][data[0].length];
        this.labels = new int[splitPoint];
        this.validate = new double[dataLabels.length-splitPoint][data[0].length];
        this.validateLabels = new int[dataLabels.length-splitPoint];
        for (int i =0;i<splitPoint;++i) {
            this.learn[i] = data[i];
            this.labels[i] = dataLabels[i];
        }

        for (int i = splitPoint;i<data.length;++i) {
            this.validate[i -splitPoint] = data[i];
            this.validateLabels[i-splitPoint] = dataLabels[i];
        }

    }

    public double fitNetwork(double[][] test, int[] testLabels) {


        double currentKappa = -1;
        int[] valPred = predict(validate);
        double newKappa = kappa(validateLabels, valPred);
//        System.out.println(String.format("Random start kappa: %f", newKappa));

        for (int i = 0; i < learn.length; ++i) {
            stochasticStep(learn[i], labels[i]);
        }

//        for (Layer layer : layers) {
//            layer.decreaseGamma();
//        }

        valPred = predict(validate);
        newKappa = kappa(validateLabels, valPred);
//        System.out.println(String.format("Start kappa: %f", newKappa));
        do {
            for (Layer layer : layers) {
                layer.backupWeights();
            }
            currentKappa = newKappa;
            for (int i = 0; i < 2*learn.length; ++i) {
                int observationId = random.nextInt(learn.length);
                stochasticStep(learn[observationId], labels[observationId]);
            }
            valPred = predict(validate);
            newKappa = kappa(validateLabels, valPred);
//            System.out.println(String.format("step done: new kappa %f", newKappa));
//            for (Layer layer : layers) {
//                layer.decreaseGamma();
//            }
        } while (newKappa > currentKappa + eps);
        for (Layer layer : layers) {
            layer.restoreWeights();
        }
        int[] pred = predict(test);
        testKappa = kappa(testLabels,pred);//currentKappa;
        return testKappa;
    }

//    def kappa(real, predicted):
//    counts = np.zeros((3, 3))
//            for i in range(real.size):
//    counts[real[i], predicted[i]] += 1
//    total = counts.sum()
//    pra = counts[0, 0] + counts[1, 1] + counts[2, 2]
//    pra = pra / total
//            pre = (counts.sum(0) / total) * (counts.sum(1) / total)
//    pre = pre.sum()
//            return (pra - pre) / (1 - pre)

    public double kappa(int[] real, int[] predict) {
        double[][] counts = new double[3][3];
        for (int i = 0; i < real.length; ++i) {
            counts[real[i]][predict[i]]++;
        }
        double total = 0;
        for (int i = 0; i < counts.length; ++i)
            for (int j = 0; j < counts[i].length; ++j)
                total += counts[i][j];

        double pra = counts[0][0] + counts[1][1] + counts[2][2];
        pra /= total;
        double[] rowSum = new double[3];
        double[] colSum = new double[3];
        for (int i = 0; i < counts.length; ++i) {
            for (int j = 0; j < counts[i].length; ++j) {
                colSum[i] += counts[i][j];
                rowSum[j] += counts[i][j];
            }
        }
        double pre = 0;
        for (int i = 0; i < colSum.length; ++i) {
            pre += colSum[i] * rowSum[i] / (total * total);
        }
        return (pra - pre) / (1 - pre);
    }

    public int[] predict(double[][] observations) {
        int[] answer = new int[observations.length];
        for (int i = 0; i < observations.length; ++i) {
            answer[i] = predict(observations[i]) > threshold ? 0 : 1;
        }
        return answer;
    }

}


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
//        System