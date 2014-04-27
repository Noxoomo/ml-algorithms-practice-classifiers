import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class DataFrameTest {
    Random rand = new Random();

    @Test
    public void testArgsort() throws Exception {


        for (int k = 0; k < 20; ++k) {
            int featuresCount = 50;
            int observationsCount = 100000 + rand.nextInt(1000);
            double[][] data = new double[featuresCount][observationsCount];
            int[] labels = new int[observationsCount];

            for (int i = 0; i < featuresCount; ++i) {
                for (int j = 0; j < observationsCount; ++j) {
                    data[i][j] = rand.nextInt();
                }
            }
            DataFrame df = new DataFrame(data,labels);

            for (int i=0;i < featuresCount; ++i) {
                int [] order = df.argsort(i);
                for (int j =0; j < order.length - 1;++j) {
                    assertTrue(df.get(i,j,order) <= df.get(i,j+1,order));
                }
            }
        }

    }

    @Test
    public void testBootstapArgsort() throws Exception {


        for (int k = 0; k < 50; ++k) {
            int featuresCount = 10;
            int observationsCount = 100000 + rand.nextInt(1000);
            double[][] data = new double[featuresCount][observationsCount];
            int[] labels = new int[observationsCount];

            for (int i = 0; i < featuresCount; ++i) {
                for (int j = 0; j < observationsCount; ++j) {
                    data[i][j] = rand.nextInt();
                }
            }
            DataFrame df = new DataFrame(data,labels);
            int[] index = new int [observationsCount];
            for (int i=0;i < index.length;++i) {
                index[i] =  rand.nextInt(observationsCount);
            }

            for (int i=0;i < featuresCount; ++i) {
                int [] order = df.argsort(i,index);
                for (int j =0; j < order.length - 1;++j) {
                    assertTrue(df.get(i,j,order) <= df.get(i,j+1,order));
                }
            }
        }

    }


}