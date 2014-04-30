package SVM;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.util.Locale;
import java.util.Scanner;

/**
 * User: Vasily
 * Date: 04.12.13
 * Time: 22:16
 */

public class GuessThirds {
    public static void main(String[] args) {
        int cols = 42;
        int rows = 11000;// 11119;//5000;//11119;
        int rowsThird = 56;
        int testRows = 1346;
        int firstLearnCount = 500;

        double learn1[][] = new double[rows][cols];
        double learn2[][] = new double[rowsThird][cols];
        //double classes[] = new double[rows];
        double test[][] = new double[testRows][cols];
        try {
            Scanner scanner = new Scanner(new FileInputStream("learnData1")).useLocale(Locale.ENGLISH);
            for (int i=0;i<rows; ++ i)
                for (int j=0;j<cols; ++j) {
                    learn1[i][j] = scanner.nextDouble();
                }
            scanner.close();
            scanner = new Scanner(new FileInputStream("learnData2")).useLocale(Locale.ENGLISH);
            for (int i=0;i<rowsThird; ++ i)
                for (int j=0;j<cols; ++j) {
                    learn2[i][j] = scanner.nextDouble();
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
        for (int i=0;i<(int) (rows / firstLearnCount); ++i) {
            double learn[][] = new double[rowsThird+firstLearnCount][cols];
            double classes[] = new double[rowsThird+firstLearnCount];
            for (int j=0;j<rowsThird;++j) {
                learn[j]=learn2[j];
                classes[j] = 1;
            }
            for (int j=0;j<firstLearnCount;++j) {
                learn[j+rowsThird] = learn1[firstLearnCount*i + j];
                classes[j+rowsThird] = -1;
            }
            RadialBasisSvm svm = new RadialBasisSvm(learn1,classes,1000);
            svm.setGamma(1.0/42);
            svm.learn();
            double curPredict[] = svm.predict(test);
            for (int k=0;k<curPredict.length;++k) {
                predict[k]+=curPredict[k];
            }

            SVM svm2 = new SVM(learn1,classes,1000);
            svm.learn();
            curPredict = svm2.predict(test);
            for (int k=0;k<curPredict.length;++k) {
                predict[k]+=0.5*curPredict[k];
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
