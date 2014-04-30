package NeuralNet;


import java.io.*;
import java.util.Locale;
import java.util.Scanner;
import java.util.concurrent.ForkJoinPool;

/**
 * User: Vasily
 * Date: 29.04.14
 * Time: 10:14
 */
public class NeuralClassifier {

    public static void main(String[] args) {
        String learnPath = "learn_transformed";
        String learnLabelsPath = "learn_transformed_answers";
        String testPath = "test_transformed";
        String valPath = "val_transformed";
        String valLabelsPath = "val_transformed_answers";
        String testLabelsPath = "test_transformed_answers";

        try {
            //read learn
            Scanner learnScanner = new Scanner(new BufferedReader(new FileReader(learnPath))).useLocale(Locale.US);
            Scanner labelsScanner = new Scanner(new BufferedReader(new FileReader(learnLabelsPath))).useLocale(Locale.US);

            int observations = learnScanner.nextInt();
            int features = learnScanner.nextInt();
            if (observations != labelsScanner.nextInt()) {
                System.err.println("File format error in learn\n");
                System.exit(0);
            }

            double[][] learn = new double[observations][features];
            int[] labels = new int[observations];
            for (int observation = 0; observation < observations; ++observation) {
                for (int feature = 0; feature < features; ++feature)
                    learn[observation][feature] = learnScanner.nextDouble();
                labels[observation] = (int) labelsScanner.nextDouble();
            }


            Scanner valScanner = new Scanner(new BufferedReader(new FileReader(valPath))).useLocale(Locale.US);
            Scanner valLabelsScanner = new Scanner(new BufferedReader(new FileReader(valLabelsPath))).useLocale(Locale.US);
            int valObservations = valScanner.nextInt();
            valScanner.nextInt();
            valLabelsScanner.nextInt();
            double[][] validate = new double[valObservations][features];
            int[] validateLabels = new int[valObservations];


            for (int observation = 0; observation < valObservations; ++observation) {
                for (int feature = 0; feature < features; ++feature)
                    validate[observation][feature] = valScanner.nextDouble();
                validateLabels[observation] = (int) valLabelsScanner.nextDouble();
            }
            //read test
            Scanner testScanner = new Scanner(new BufferedReader(new FileReader(testPath))).useLocale(Locale.US);
            Scanner testLabelsScanner = new Scanner(new BufferedReader(new FileReader(testLabelsPath))).useLocale(Locale.US);

            int testObservations = testScanner.nextInt();
            double[][] test = new double[testObservations][features];
            int[] testLabels = new int[testObservations];
            testLabelsScanner.nextInt();

            if (testScanner.nextInt() != features) {
                System.err.println("File format error in test\n");
                System.exit(0);
            }
            for (int observation = 0; observation < testObservations; ++observation) {
                for (int feature = 0; feature < features; ++feature) {
                    test[observation][feature] = testScanner.nextDouble();      //!!! In test test[i] — i'th observation instead of i's list of features
                }
                testLabels[observation] = (int) testLabelsScanner.nextDouble();
            }

            testScanner.close();
            learnScanner.close();
            labelsScanner.close();
            testLabelsScanner.close();
            valScanner.close();
            valLabelsScanner.close();

            long startTime = System.currentTimeMillis();
            System.out.println("Start fitting Neural Net");

            int[] topology = new int[]{features, 150,150, 1};
            NetworkChooser chooser = new NetworkChooser(topology,learn,labels,validate,validateLabels,32);
            ForkJoinPool pool = new ForkJoinPool(4);
            NeuralNet bestNetwork = pool.invoke(chooser);

//            System.out.println("First try");
//            NeuralNet bestNetwork = new NeuralNet(topology);
//            double bestKappa = bestNetwork.fitNetwork(learn, labels, validate, validateLabels);
////            double bestKappa = bestNetwork.fitNetwork(learn, labels, learn,labels);
//
//            for (int i = 0; i < 100; ++i) {
//                System.out.println("Next try");
//                NeuralNet net = new NeuralNet(topology);
//                double netKappa = net.fitNetwork(learn, labels, validate, validateLabels);
////                double netKappa = net.fitNetwork(learn, labels, learn,labels);
//                if (netKappa > bestKappa) {
//                    bestKappa = netKappa;
//                    bestNetwork = net;
//                }
//            }


            System.out.println(String.format("\n\nWorking time: %d\nValidate kappa: %f", (System.currentTimeMillis() - startTime) / 60000, bestNetwork.testKappa));

            int[] predictLabels = bestNetwork.predict(test);
            double testKappa = bestNetwork.kappa(testLabels, predictLabels);
            System.out.println(String.format("Test kappa: %f", testKappa));
            BufferedWriter writer = new BufferedWriter(new FileWriter("result"));
            for (int label : predictLabels) {
                writer.write(String.format("%d\n", label));
            }
            writer.flush();
            writer.close();
        } catch (IOException e) {
            System.err.println("IO exception");
        }

    }


}
