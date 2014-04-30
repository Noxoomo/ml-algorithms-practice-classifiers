package SVM;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.util.Locale;
import java.util.Scanner;

/**
 * User: Vasily
 * Date: 30.11.13
 * Time: 9:39
 */
public class MoreThanOneSvm {
    public static void main(String[] args) {
        int cols = 42;
        int rows =  11000;//5000;//11119;
        int testRows = 1346;
        int models  = 500;

        double learn[][] = new double[rows][cols];
        double classes[] = new double[rows];
        double test[][] = new double[testRows][cols];
        try {
            Scanner scanner = new Scanner(new FileInputStream("learnData")).useLocale(Locale.ENGLISH);
            for (int i=0;i<rows; ++ i)
                for (int j=0;j<cols; ++j) {
                    learn[i][j] = scanner.nextDouble();
                }
            scanner.close();
            scanner = new Scanner(new FileInputStream("learnClasses")).useLocale(Locale.ENGLISH);
            for (int i=0;i<rows;++i) {
                classes[i] = scanner.nextDouble();
            }
            scanner.close();
            scanner = new Scanner(new FileInputStream("testData")).useLocale(Locale.ENGLISH);
            for (int i=0;i<testRows; ++ i)
                for (int j=0;j<cols; ++j) {
                    test[i][j] = scanner.nextDouble();
                }

        } catch (Exception e) {
            System.out.print("error");
            return;
        }

        long start = System.currentTimeMillis();
        double predict[] = new double[testRows];
        int svmRows = rows / models;
        double svmLearn[][] = new double[svmRows][cols];
        double svmClasses[] = new double[svmRows];
        for (int i=0;i< models;++i) {
            for (int j=0;j<svmRows;++j) {
                svmLearn[j] = learn[i*svmRows+j];
                svmClasses[j] = classes[j+svmRows*i];
            }
            RadialBasisSvm svm = new RadialBasisSvm(svmLearn,svmClasses,1000);
            svm.setGamma(1.0/42);
            svm.learn();
            double svmPredict[] = svm.predict(test);
            for (int k=0;k<svmPredict.length;++k) {
                predict[k] += svmPredict[k];
            }
        }

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter("predict"));
            for (int i=0;i<predict.length;++i) {
                String line = String.valueOf(predict[i]) + "\n";
                writer.write(line);
            }
            writer.close();
        } catch (Exception e) {

        }

        System.err.print("Working time: ");
        System.err.println(System.currentTimeMillis()-start);
    }
}
