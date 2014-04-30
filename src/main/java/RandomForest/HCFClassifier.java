package RandomForest;

import java.io.*;
import java.util.Locale;
import java.util.Scanner;

/**
 * User: Vasily
 * Date: 27.04.14
 * Time: 13:57
 */

public class HCFClassifier {
    public static void main(String[] args) {
        String learnPath = "learn_transformed";
        String learnLabelsPath = "learn_transformed_answers";
        String testPath = "test_transformed";
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
            double[][] learn = new double[features][observations];
            int[] labels = new int[observations];
            for (int observation = 0; observation < observations; ++observation) {
                for (int feature = 0; feature < features; ++feature)
                    learn[feature][observation] = learnScanner.nextDouble();
                labels[observation] = (int) labelsScanner.nextDouble();
            }

            //read test
            Scanner testScanner = new Scanner(new BufferedReader(new FileReader(testPath))).useLocale(Locale.US);
            int testObservations = testScanner.nextInt();
            double[][] test = new double[testObservations][features];

            if (testScanner.nextInt() != features) {
                System.err.println("File format error in test\n");
                System.exit(0);
            }
            for (int observation = 0; observation < testObservations; ++observation) {
                for (int feature = 0; feature < features; ++feature) {
                    test[observation][feature] = testScanner.nextDouble();      //!!! In test test[i] — i'th observation instead of i's list of features
                }
            }

            testScanner.close();
            learnScanner.close();
            labelsScanner.close();

            long startTime = System.currentTimeMillis();
            System.out.println("Start fitting forest");
            RandomForestClassifier classifier = new RandomForestClassifier(learn, labels, 7, 3);
            classifier.addTrees(6000);
            int[] predictLabels = classifier.predict(test);
            System.out.println(String.format("Working time: %d",(System.currentTimeMillis() - startTime) / 60000));
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
