package SVM;

import java.io.FileInputStream;
import java.util.Locale;
import java.util.Scanner;

/**
 * User: Vasily
 * Date: 07.12.13
 * Time: 19:36
 */
public class FindGamma {
    public static void main(String[] args) {
        int cols = 42;
        int rows = 1000;//5000;//11119;
        int testRows = 1346;

        double learn[][] = new double[rows][cols];
        double classes[] = new double[rows];
        double test[][] = new double[testRows][cols];
        try {
            Scanner scanner = new Scanner(new FileInputStream("learnData")).useLocale(Locale.ENGLISH);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j) {
                    learn[i][j] = scanner.nextDouble();
                }
            scanner.close();
            scanner = new Scanner(new FileInputStream("learnClasses")).useLocale(Locale.ENGLISH);
            for (int i = 0; i < rows; ++i) {
                classes[i] = scanner.nextDouble();
            }
            scanner.close();
            scanner = new Scanner(new FileInputStream("testData")).useLocale(Locale.ENGLISH);
            for (int i = 0; i < testRows; ++i)
                for (int j = 0; j < cols; ++j) {
                    test[i][j] = scanner.nextDouble();
                }

        } catch (Exception e) {
            System.out.print("error");
            return;
        }

        long start = System.currentTimeMillis();
        double gamma = 0.0001;
        double step = 0.0001;
        double best = 0.02;
        double bestRate = 0;
        int toLearn = 200;
        double X[][] = new double[toLearn][cols];
        double y[] = new double[toLearn];
        double validate[][] = new double[rows - toLearn][cols];
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) {
                if (i < toLearn) {
                    X[i][j] = learn[i][j];
                    y[i] = classes[i];
                } else {
                    validate[i - toLearn][j] = learn[i][j];
                }
            }


        RadialBasisSvm sv = new RadialBasisSvm(X,y,1000);
        sv.setGamma(0.0017);
        sv.learn();
        while (gamma < 0.02) {
            RadialBasisSvm svm = new RadialBasisSvm(X, y, 10000);
            svm.setGamma(gamma);
            svm.learn();
            double predict[] = svm.predict(validate);
            double count = 0;
            for (int i = toLearn; i < classes.length; ++i) {
                if (predict[i - toLearn] == classes[i]) {
                    count++;
                }
            }
            if (count / (classes.length - toLearn) > bestRate) {
                bestRate = count / (classes.length - toLearn);
                best = gamma;
            }
            gamma += step;

        }
        System.err.print("Working time: ");
        System.err.println(System.currentTimeMillis() - start);
        System.err.println(best);
    }
}
