package SVM;

import SVM.SVMAbstract;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.util.Locale;
import java.util.Scanner;

/**
 * User: Vasily
 * Date: 08.12.13
 * Time: 13:08
 */
public class ThreeSvm {
    public static void main(String[] args) {
        int cols = 42;
        int rows = 11000;
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

        SVMAbstract svm = new RadialBasisSvm(learn, classes, 1);
        ((RadialBasisSvm) svm).setGamma(0.025);
        ((RadialBasisSvm) svm).learn();
        double predict[] = svm.predict(test);

        svm = new PolynomialSvm(learn, classes, 1);
        ((PolynomialSvm) svm).setCoef(1);
        ((PolynomialSvm) svm).setDegree(2);
        ((PolynomialSvm) svm).setGamma(0.1);
        ((PolynomialSvm) svm).learn();
        double predict2[] = svm.predict(test);

//        svm = new SigmoidKernelSvm(learn, classes, 1);
//        ((SigmoidKernelSvm) svm).setCoef(-1);
//        ((SigmoidKernelSvm) svm).setGamma(0.002);
//        ((SigmoidKernelSvm) svm).learn();
//        double predict3[] = svm.predict(test);
        double X[][] = new double[3000][cols];
        double y[] = new double[3000];
        for (int i = 0; i < 3000; ++i)
            for (int j = 0; j < cols; ++j) {
                if (i < 3000) {
                    X[i][j] = learn[i][j];
                    y[i] = classes[i];
                }
            }
        svm = new MinKernel(X, y, 10);
        ((MinKernel)svm).learn();
        double predict4[] = svm.predict(test);

        for (int i = 0; i < predict2.length; ++i) {
            predict[i] += predict2[i];
            predict[i] += predict4[i];
        }
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter("predict"));
            for (int i = 0; i < predict.length; ++i) {
                String line = String.valueOf(predict[i]) + "\n";
                writer.write(line);
            }
            writer.close();
        } catch (Exception e) {

        }

        System.err.print("Working time: ");
        System.err.println(System.currentTimeMillis() - start);
    }
}
